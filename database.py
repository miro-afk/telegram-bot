# database.py
import psycopg2
from psycopg2 import sql
from config import DATABASE_CONFIG
import random

def get_db_connection():
    return psycopg2.connect(**DATABASE_CONFIG)

def get_random_product():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT id, name, description, price, image_url FROM products ORDER BY RANDOM() LIMIT 1")
        product = cursor.fetchone()
        
        if product:
            return {
                'id': product[0],
                'name': product[1],
                'description': product[2],
                'price': product[3],
                'image_url': product[4]
            }
        return None
    finally:
        cursor.close()
        conn.close()

def get_product_by_category(category):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT id, name, description, price, image_url FROM products WHERE category = %s",
            (category,)
        )
        products = cursor.fetchall()
        choice = random.choice(products)

        if choice:
            return {
                'id': choice[0],
                'name': choice[1],
                'description': choice[2],
                'price': choice[3],
                'image_url': choice[4]
            }
    finally:
        cursor.close()
        conn.close()