from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
import io, os, shutil
import pandas as pd
from werkzeug.utils import secure_filename
from passlib.hash import bcrypt
import os
import pandas as pd
import io
from datetime import datetime, timedelta, timezone
import utilities as ut
from models import ensure_schema
from utils import (
    add_user, get_user, get_all_users, update_user_status,
    set_tier_and_activate, zip_user_folder, admin_replace_dataset, 
    reset_password, update_payment, check_payment_status
)
from AI import model_work as mt
from AI.trend_score_compute import get_google_trend

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change to a strong secret in production

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure database schema
ensure_schema()

# -----------------------------
# Entry page: redirect to login or dashboard
# -----------------------------
@app.route("/", methods=["GET"])
def entry_page():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

# -----------------------------
# Login route
# -----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username").lower()
        password = request.form.get("password")

        # Admin login
        if username == "admin" and password == "admin123":
            session["username"] = "admin"
            session["is_admin"] = True
            flash("Logged in as Admin", "success")
            return redirect(url_for("admin_dashboard"))

        # Normal user login
        user = get_user(username)
        if user and bcrypt.verify(password, user.password):
            if user.status == "pending":
                flash("Account pending admin approval.", "warning")
            elif user.status == "paused":
                flash("Your account has been paused by the admin.", "danger")
            else:
                session["username"] = user.username
                session["is_admin"] = False
                flash(f"Welcome {username}!", "success")
                return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password", "danger")

    return render_template("login.html")

# -----------------------------
# Signup route
# -----------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "username" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username").lower()
        password = request.form.get("password")
        uploaded_file = request.files.get("dataset")

        if username == "admin":
            flash("Cannot sign up as admin!", "danger")
            return redirect(url_for("signup"))

        if username and password:
            if get_user(username):
                flash("Username already exists", "danger")
            else:
                add_user(username, password)
                flash("Account created. Waiting for admin approval.", "success")

                # Save uploaded dataset if exists
                if uploaded_file and uploaded_file.filename != '':
                    filename = secure_filename(uploaded_file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    uploaded_file.save(file_path)
                    df = pd.read_csv(file_path)
                    admin_replace_dataset(username, df)
        else:
            flash("Fill all fields", "warning")

    return render_template("signup.html")

# -----------------------------
# Logout route
# -----------------------------
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))

# -----------------------------
# Dashboard route (user)
# -----------------------------
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "username" not in session:
        flash("Please login first", "warning")
        return redirect(url_for("login"))

    # Admin goes to admin_dashboard
    if session.get("is_admin"):
        return redirect(url_for("admin_dashboard"))

    # Normal user dashboard
    user = get_user(session["username"])
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("login"))
    
    page = request.args.get("page", "views")
    
    # Check payment status - FIXED timezone comparison
    payment_ok = check_payment_status(user)

    tiers = [
        {"name":"Tier 1: Views Predictor","desc":"Predict how much views a video can have in its lifetime uploaded on a certain date","price":"$50"},
        {"name":"Tier 2: Trending & Similar","desc":"Find the best trending movies/series for a month, get similar movies, plus Tier 1 features","price":"$100"},
        {"name":"Tier 3: Upload Calendar","desc":"Generate a complete list of what to upload daily monthly, plus Tier 2 features","price":"$150"},
        {"name":"Tier 4: Smart Chatbot","desc":"Chatbot understands your upload history and provides insights, plus Tier 3 features","price":"$200"},
        {"name":"Tier 5: Future Premium","desc":"All features up to Tier 4, plus more (details TBD)","price":"$300"},
    ]

    tier_label = {0:"None",1:"Tier 1",2:"Tier 2",3:"Tier 3",4:"Tier 4",5:"Tier 5"}

    if request.method == "POST":
        action = request.form.get("action")

        # Password reset
        if action == "reset_password":
            new_pass = request.form.get("new_password")
            if new_pass:
                reset_password(user.id, new_pass)
                flash("Password updated!", "success")
            return redirect(url_for("dashboard", page="account"))

        # Prediction
        elif action == "predict":
            if not payment_ok or user.payment_tier not in [1, 3]:
                flash("Your tier does not include this feature. Please contact admin.", "warning")
                return redirect(url_for("dashboard", page="views"))
                
            movie_name = request.form.get("movie_name")
            release_date = request.form.get("release_date")
            
            try:
                # Check if user has a dataset
                user_dir = f"User/{user.username}"
                if not os.path.exists(user_dir) or not os.path.exists(f"{user_dir}/{user.username}.csv"):
                    flash("No dataset found. Please contact admin to upload your dataset.", "warning")
                    return redirect(url_for("dashboard", page="views"))
                
                # Check if model needs training
                model_path = f"{user_dir}/model.pth"
                if not os.path.exists(model_path):
                    flash("Training model for the first time. This may take a few minutes...", "info")
                    mt.model_train(user_dir, f"{user.username}.csv")
                
                # Initialize cache
                cache_obj = ut.cache_memory(user.username)
                cache_obj.check_for_cache()
                loaded_data = cache_obj.loaded_dataframe

                # Check cache first
                mask = (loaded_data["Title"].str.lower() == movie_name.lower()) & (loaded_data["Upload_Date"] == release_date)
                found_data = loaded_data.loc[mask]
                
                if not found_data.empty:
                    row = found_data.iloc[0]
                    results = {
                        "title": row["Title"],
                        "release_date": row["Upload_Date"],
                        "hype_score": row["Hype_Score"],
                        "min": row["Min"],
                        "max": row["Max"],
                    }
                    flash("Prediction loaded from cache!", "info")
                else:
                    # Make prediction
                    results = mt.model_inference(movie_name, release_date, user_dir, user.username)
                    
                    if results:
                        # Save to cache
                        cache_obj.dump_data(
                            results["title"], results["release_date"], results["hype_score"], results["min"], results["max"]
                        )
                        flash("Prediction successful!", "success")
                    else:
                        flash("Prediction failed. Please try again.", "danger")
                        return redirect(url_for("dashboard", page="views"))
                
                session["last_result"] = results
            except Exception as e:
                flash(f"Prediction error: {str(e)}", "danger")

            return redirect(url_for("dashboard", page="views"))
        
        # View prediction history
        elif action == "view_history":
            session["show_history"] = True
            return redirect(url_for("dashboard", page="views"))
        
        # Close history view
        elif action == "close_history":
            session["show_history"] = False
            return redirect(url_for("dashboard", page="views"))
        
        # Train model
        elif action == "train_model":
            try:
                user_dir = f"User/{user.username}"
                mt.model_train(user_dir, f"{user.username}.csv")
                flash("Model trained successfully!", "success")
            except Exception as e:
                flash(f"Model training failed: {str(e)}", "danger")
            return redirect(url_for("dashboard", page="views"))

    # Get prediction history for display
    prediction_history = None
    if page == "views" and session.get("show_history"):
        try:
            cache_obj = ut.cache_memory(user.username)
            cache_obj.check_for_cache()
            prediction_history = cache_obj.loaded_dataframe
        except:
            prediction_history = pd.DataFrame()

    return render_template("dashboard.html", user=user, page=page, payment_ok=payment_ok, 
                         tiers=tiers, tier_label=tier_label, last_result=session.get("last_result"),
                         prediction_history=prediction_history, show_history=session.get("show_history", False))

# -----------------------------
# Admin dashboard route
# -----------------------------
@app.route("/admin", methods=["GET", "POST"])
def admin_dashboard():
    if "username" not in session or not session.get("is_admin"):
        flash("Admin access required.", "danger")
        return redirect(url_for("login"))

    users = get_all_users()

    if request.method == "POST":
        action = request.form.get("action")
        user_id = int(request.form.get("user_id"))
        user = next((u for u in users if u.id == user_id), None)

        if not user:
            flash("User not found.", "danger")
            return redirect(url_for("admin_dashboard"))

        if action == "approve":
            update_user_status(user_id, "active")
            flash(f"{user.username} approved!", "success")

        elif action == "pause":
            update_user_status(user_id, "paused")
            flash(f"{user.username} paused.", "warning")

        elif action == "reactivate":
            update_user_status(user_id, "active")
            flash(f"{user.username} reactivated.", "success")

        elif action == "set_tier":
            tier = int(request.form.get("tier"))
            set_tier_and_activate(user_id, tier)
            flash(f"{user.username} set to Tier {tier} and activated.", "success")

        elif action == "download_zip":
            zip_bytes = zip_user_folder(user.username)
            if zip_bytes:
                return send_file(
                    io.BytesIO(zip_bytes),
                    mimetype="application/zip",
                    as_attachment=True,
                    download_name=f"{user.username}_bundle.zip"
                )
            else:
                flash("No user folder found to zip.", "warning")

        elif action == "upload_dataset":
            file = request.files.get("dataset")
            if file and file.filename.endswith(".csv"):
                try:
                    df = pd.read_csv(file)
                    user_dir = f"User/{user.username}"

                    # Remove old folder if exists
                    if os.path.exists(user_dir):
                        shutil.rmtree(user_dir)

                    # Save new dataset
                    admin_replace_dataset(user.username, df)
                    flash("Dataset updated successfully.", "success")
                except Exception as e:
                    flash(f"Failed to update dataset: {e}", "danger")
            else:
                flash("Please upload a valid CSV file.", "warning")

        elif action == "retrain_model":
            try:
                user_dir = f"User/{user.username}"
                if os.path.exists(f"{user_dir}/{user.username}.csv"):
                    mt.model_train(user_dir, f"{user.username}.csv")
                    flash("Model retrained successfully.", "success")
                else:
                    flash("No dataset found for this user.", "warning")
            except Exception as e:
                flash(f"Retrain failed: {e}", "danger")

        return redirect(url_for("admin_dashboard"))

    # Filter users
    pending_users = [u for u in users if u.status == "pending" and u.username.lower() != "admin"]
    active_users = [u for u in users if u.status != "pending" and u.username.lower() != "admin"]

    tier_label = {0: "None", 1: "Tier 1 (Views)", 2: "Tier 2 (Similar)", 3: "Tier 3 (Both)"}

    return render_template(
        "admin.html",
        pending_users=pending_users,
        active_users=active_users,
        tier_label=tier_label,
    )

# -----------------------------
# Download prediction history
# -----------------------------
@app.route("/download_history")
def download_history():
    if "username" not in session:
        flash("Please login first", "warning")
        return redirect(url_for("login"))
    
    user = get_user(session["username"])
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("login"))
    
    try:
        cache_obj = ut.cache_memory(user.username)
        cache_obj.check_for_cache()
        history_df = cache_obj.loaded_dataframe
        
        # Convert DataFrame to CSV
        csv_data = history_df.to_csv(index=False)
        return send_file(
            io.BytesIO(csv_data.encode('utf-8')),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"{user.username}_prediction_history.csv"
        )
    except Exception as e:
        flash(f"Failed to download history: {str(e)}", "danger")
        return redirect(url_for("dashboard", page="views"))

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)