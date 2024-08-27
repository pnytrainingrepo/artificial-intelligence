import streamlit as st
import requests

# Define the base URL of your FastAPI server
BASE_URL = "http://127.0.0.1:8000"

# Function to register a new user
def register_user(username, password):
    data = {
        "username": username,
        "password": password
    }
    response = requests.post(f"{BASE_URL}/users/", json=data)
    return response

# Function to log in and get a token
def login_user(username, password):
    data = {
        "username": username,
        "password": password
    }
    response = requests.post(f"{BASE_URL}/token", data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        return None

# Function to get current user information
def get_current_user(token):
    headers = {
        "Authorization": f"Bearer {token}"
    }
    response = requests.get(f"{BASE_URL}/users/me/", headers=headers)
    return response.json()

# Function to fetch all products
def fetch_products():
    response = requests.get(f"{BASE_URL}/products/")
    return response.json()

# Function to fetch all orders
def fetch_orders():
    response = requests.get(f"{BASE_URL}/orders/")
    return response.json()


# Function to create a new product (authentication required)
def create_product(token, product_name, product_desc, product_price, product_qty):
    headers = {
        "Authorization": f"Bearer {token}"
    }
    data = {
        "product_name": product_name,
        "product_desc": product_desc,
        "product_price": product_price,
        "product_qty": product_qty
    }
    response = requests.post(f"{BASE_URL}/products/", json=data, headers=headers)
    return response

# Function to place an order (authentication required)
def place_order(token, product_id, order_qty):
    headers = {
        "Authorization": f"Bearer {token}"
    }
    data = {
        "product_id": product_id,
        "order_qty": order_qty
    }
    response = requests.post(f"{BASE_URL}/orders/", json=data, headers=headers)
    return response

# Streamlit app
st.title("Mart API Frontend")

menu = ["Register", "Login", "View Products", "Create Product", "Place Order", "View Orders"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register":
    st.subheader("Register a New User")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        response = register_user(username, password)
        if response.status_code == 200:
            st.success("User registered successfully!")
        else:
            st.error("Error occurred during registration")

elif choice == "Login":
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        token = login_user(username, password)
        if token:
            st.session_state["token"] = token
            st.success("Logged in successfully!")
        else:
            st.error("Login failed!")

elif choice == "View Products":
    st.subheader("Available Products")
    products = fetch_products()
    for product in products:
        st.write(f"ID: {product['product_id']} - {product['product_name']} - {product['product_price']} USD - Qty: {product['product_qty']}")

elif choice == "Create Product":
    st.subheader("Create a New Product")
    if "token" in st.session_state:
        product_name = st.text_input("Product Name")
        product_desc = st.text_input("Product Description")
        product_price = st.number_input("Product Price", min_value=0.0)
        product_qty = st.number_input("Product Quantity", min_value=0)

        if st.button("Create Product"):
            response = create_product(st.session_state["token"], product_name, product_desc, product_price, product_qty)
            if response.status_code == 200:
                st.success("Product created successfully!")
            else:
                st.error("Failed to create product")
    else:
        st.error("You need to be logged in to create a product.")
        
elif choice == "View Orders":
    st.subheader("Available Orders")
    orders = fetch_orders()
    for order in orders:
        st.write(f"ID: {order['order_id']} - Product_ID: {order['product_id']} -  Qty: {order['order_qty']}")

elif choice == "Place Order":
    st.subheader("Place an Order")
    if "token" in st.session_state:
        product_id = st.number_input("Product ID", min_value=1)
        order_qty = st.number_input("Order Quantity", min_value=1)

        if st.button("Place Order"):
            response = place_order(st.session_state["token"], product_id, order_qty)
            if response.status_code == 200:
                st.success("Order placed successfully!")
            else:
                st.error("Failed to place order")
    else:
        st.error("You need to be logged in to place an order.")
