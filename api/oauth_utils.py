import requests

def get_user_info_from_google(id_token):
    """
    Validate Google ID token using tokeninfo endpoint.
    Returns user info dict if valid, else None.
    """
    resp = requests.get(
        f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
    )

    if resp.status_code == 200:
        return resp.json()

    return None
