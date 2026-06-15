from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from ..extensions import db
from ..models import User

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.get("/register")
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.dashboard"))
    return render_template("register.html")


@auth_bp.post("/register")
def register_post():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.dashboard"))

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    phone = request.form.get("phone", "").strip()
    caregiver_email = request.form.get("caregiver_email", "").strip().lower() or None
    caregiver_phone = request.form.get("caregiver_phone", "").strip() or None

    if not name or not email or not password:
        flash("Name, email, and password are required.", "error")
        return render_template("register.html"), 400

    if User.query.filter_by(email=email).first():
        flash("That email is already registered.", "error")
        return render_template("register.html"), 400

    user = User(
        name=name,
        email=email,
        phone=phone,
        caregiver_email=caregiver_email,
        caregiver_phone=caregiver_phone,
    )
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    flash("Account created. Please log in.", "success")
    return redirect(url_for("auth.login"))


@auth_bp.get("/login")
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.dashboard"))
    return render_template("login.html")


@auth_bp.post("/login")
def login_post():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.dashboard"))

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    user = User.query.filter_by(email=email).first()

    if not user or not user.check_password(password):
        flash("Invalid email or password.", "error")
        return render_template("login.html"), 401

    login_user(user)
    return redirect(url_for("dashboard.dashboard"))


@auth_bp.post("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "success")
    return redirect(url_for("auth.login"))
