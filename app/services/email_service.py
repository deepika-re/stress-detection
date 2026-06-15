import smtplib
from email.mime.text import MIMEText

from flask import current_app


def send_stress_alert(user, stress_event, reading):
    username = current_app.config.get("MAIL_USERNAME")
    password = current_app.config.get("MAIL_PASSWORD")
    sender = current_app.config.get("MAIL_DEFAULT_SENDER")

    recipients = [email for email in [user.email, user.caregiver_email] if email]
    if not recipients:
        return {
            "status": "skipped",
            "recipient_email": None,
            "message": "No recipient email configured",
        }

    if not username or not password or not sender:
        return {
            "status": "skipped",
            "recipient_email": ", ".join(recipients),
            "message": "Email credentials are not configured",
        }

    body = (
        f"High stress detected for {user.name}.\n\n"
        f"Stress score: {stress_event.stress_score}\n"
        f"Heart rate: {reading.heart_rate:.1f} bpm\n"
        f"GSR: {reading.gsr:.2f}\n"
        f"Captured at: {reading.captured_at.isoformat()} UTC\n\n"
        "Please check in if this alert was unexpected."
    )
    message = MIMEText(body)
    message["Subject"] = "Stress Alert - High Stress Detected"
    message["From"] = sender
    message["To"] = ", ".join(recipients)

    try:
        with smtplib.SMTP(current_app.config["MAIL_SERVER"], current_app.config["MAIL_PORT"], timeout=10) as server:
            if current_app.config["MAIL_USE_TLS"]:
                server.starttls()
            server.login(username, password)
            server.sendmail(sender, recipients, message.as_string())
        return {
            "status": "sent",
            "recipient_email": ", ".join(recipients),
            "message": "Alert email sent",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "recipient_email": ", ".join(recipients),
            "message": str(exc),
        }
