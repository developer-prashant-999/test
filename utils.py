from models import engine, User
from sqlalchemy.orm import sessionmaker
from passlib.hash import bcrypt
from datetime import datetime, timezone,timedelta
import os, shutil, io
import pandas as pd
import utilities as ut

Session = sessionmaker(bind=engine)

def add_user(username, password):
    with Session() as session:
        hashed_password = bcrypt.hash(password)
        user = User(username=username.lower(), password=hashed_password)
        session.add(user)
        session.commit()

def get_user(username):
    with Session() as session:
        return session.query(User).filter_by(username=username.lower()).first()

def get_all_users():
    with Session() as session:
        return session.query(User).all()

def update_user_status(user_id, status):
    with Session() as session:
        user = session.query(User).filter_by(id=user_id).first()
        if user:
            user.status = status
            session.commit()

def set_tier_and_activate(user_id, tier: int):
    with Session() as session:
        user = session.query(User).filter_by(id=user_id).first()
        if user:
            user.payment_tier = tier
            user.payment_active = 1 if tier in (1, 2, 3) else 0
            user.payment_start = datetime.utcnow() if user.payment_active == 1 else None
            if user.status != "paused":
                user.status = "active"
            session.commit()

def reset_password(user_id, new_password):
    with Session() as session:
        user = session.query(User).filter_by(id=user_id).first()
        if user:
            user.password = bcrypt.hash(new_password)
            session.commit()

def zip_user_folder(username: str) -> bytes:
    folder_path = os.path.join("User", username.lower())
    if not os.path.isdir(folder_path):
        return b""
    buf = io.BytesIO()
    tmp_base = f"{username.lower()}_bundle"
    tmp_zip_path = shutil.make_archive(tmp_base, 'zip', root_dir=folder_path)
    with open(tmp_zip_path, "rb") as f:
        buf.write(f.read())
    os.remove(tmp_zip_path)
    buf.seek(0)
    return buf.read()
def update_payment(user_id, active=1):
    """Legacy toggle for payment_active; keeps payment_start in sync."""
    Session = sessionmaker(bind=engine)
    with Session() as session:
        user = session.query(User).filter_by(id=user_id).first()
        if user:
            user.payment_active = active
            user.payment_start = datetime.now(timezone.utc) if active else None
            session.commit()

# Add this function to check payment status with expiry
def check_payment_status(user):
    """Check if user's payment is active and not expired"""
    if not user.payment_active or not user.payment_start:
        return False
    
    # Make both datetimes timezone-aware for comparison
    expiry = user.payment_start.replace(tzinfo=timezone.utc) + timedelta(days=30)
    current_time = datetime.now(timezone.utc)
    
    if current_time <= expiry:
        return True
    else:
        # Auto-expire if payment is overdue
        update_payment(user.id, active=0)
        return False
    
def admin_replace_dataset(username: str, df: pd.DataFrame):
    # Use the Create_User class from utilities
    user_creator = ut.Create_User(username.lower(), df)