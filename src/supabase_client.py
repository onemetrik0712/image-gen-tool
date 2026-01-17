#!/usr/bin/env python3
"""
Supabase client for Vault feature - image storage and metadata.
"""

import os
import uuid
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client, Client

# Storage bucket name
VAULT_BUCKET = "vault-images"

# Table name
VAULT_TABLE = "vault_images"


def get_secret(key: str, fallback_env: bool = True) -> Optional[str]:
    """
    Get a secret from st.secrets (Streamlit Cloud) or .env (local dev).

    Args:
        key: The secret key to retrieve
        fallback_env: Whether to fall back to environment variables

    Returns:
        The secret value or None if not found
    """
    # Try st.secrets first (Streamlit Cloud)
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    # Fall back to environment variable
    if fallback_env:
        load_dotenv()
        return os.getenv(key)

    return None


def get_supabase_client() -> Optional[Client]:
    """
    Initialize and return Supabase client.

    Returns:
        Supabase client or None if credentials not found
    """
    url = get_secret("SUPABASE_URL")
    key = get_secret("SUPABASE_KEY")

    if not url or not key:
        return None

    return create_client(url, key)


def save_to_vault(
    image_data: bytes,
    prompt: str,
    model: str,
    cost: float,
    user_id: str = "default"
) -> dict:
    """
    Upload image to storage and save metadata to database.

    Args:
        image_data: Image bytes to upload
        prompt: The prompt used to generate the image
        model: Model name used for generation
        cost: Cost of generation
        user_id: User identifier (default for single-user mode)

    Returns:
        Dict with 'success' bool and 'data' or 'error' message
    """
    client = get_supabase_client()
    if not client:
        return {"success": False, "error": "Supabase not configured"}

    try:
        # Generate unique ID and filename
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.webp"

        # Upload image to storage
        storage_response = client.storage.from_(VAULT_BUCKET).upload(
            path=filename,
            file=image_data,
            file_options={"content-type": "image/webp"}
        )

        # Get public URL for the image
        image_url = client.storage.from_(VAULT_BUCKET).get_public_url(filename)

        # Save metadata to database
        metadata = {
            "id": image_id,
            "image_url": image_url,
            "prompt": prompt,
            "model": model,
            "cost": cost,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat()
        }

        db_response = client.table(VAULT_TABLE).insert(metadata).execute()

        return {
            "success": True,
            "data": {
                "id": image_id,
                "image_url": image_url
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def get_vault_images(user_id: str = "default") -> dict:
    """
    Retrieve all saved images with metadata from the vault.

    Args:
        user_id: User identifier to filter by

    Returns:
        Dict with 'success' bool and 'data' (list of images) or 'error'
    """
    client = get_supabase_client()
    if not client:
        return {"success": False, "error": "Supabase not configured"}

    try:
        response = client.table(VAULT_TABLE)\
            .select("*")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .execute()

        return {"success": True, "data": response.data}

    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_from_vault(image_id: str) -> dict:
    """
    Remove image from storage and metadata from database.

    Args:
        image_id: UUID of the image to delete

    Returns:
        Dict with 'success' bool and optional 'error' message
    """
    client = get_supabase_client()
    if not client:
        return {"success": False, "error": "Supabase not configured"}

    try:
        # Delete from storage
        filename = f"{image_id}.webp"
        client.storage.from_(VAULT_BUCKET).remove([filename])

        # Delete from database
        client.table(VAULT_TABLE).delete().eq("id", image_id).execute()

        return {"success": True}

    except Exception as e:
        return {"success": False, "error": str(e)}
