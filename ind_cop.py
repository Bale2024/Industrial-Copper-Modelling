import pickle
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
#Load Models and encoders
model_sp = pickle.load(open("xg_boost.pkl", "rb"))
model_sts = pickle.load(open("rf_clf.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb")) 
status_map = {0: "Lost", 1: "Won"}
#status_inv_map = {v: k for k, v in status_map.items()}

def get_user_inputs(encoders):
    # Numeric I/P
    quantity_tons = st.number_input("Quantity tons", min_value=0.0, step=0.1)
    thickness = st.number_input("Thickness", min_value=0.0, step=0.1)
    width = st.number_input("Width", min_value=0.0, step=1.0)
    delivery_date_year = st.number_input("Delivery Date Year", min_value=2000, max_value=2100, step=1)
    item_date_year = st.number_input("Item Date Year", min_value=2000, max_value=2100, step=1)
    # Categorical drop_down
    country = st.selectbox("Country", encoders['country'].classes_)
    item_type = st.selectbox("Item Type", encoders['item type'].classes_)
    application = st.selectbox("Application", encoders['application'].classes_)
    product_ref = st.selectbox("Product Reference", encoders['product_ref'].classes_)
    material_ref = st.selectbox("Material Reference", encoders['material_ref'].classes_)
    try:
        country_enc = int(encoders['country'].transform([country])[0])
        item_type_enc = int(encoders['item type'].transform([item_type])[0])
        application_enc = int(encoders['application'].transform([application])[0])
        product_ref_enc = int(encoders['product_ref'].transform([product_ref])[0])
        material_ref_enc = int(encoders['material_ref'].transform([material_ref])[0])
    except Exception as e:
        st.error(f"Encoding error: {e}")
        st.stop()
    return {
        "quantity_tons": quantity_tons,
        "thickness": thickness,
        "width": width,
        "delivery_date_year": delivery_date_year,
        "item_date_year": item_date_year,
        "country_enc": country_enc,
        "item_type_enc": item_type_enc,
        "application_enc": application_enc,
        "product_ref_enc": product_ref_enc,
        "material_ref_enc": material_ref_enc,
    }    
st.set_page_config(page_title="Industrial Copper Modelling", layout="centered")
st.title("INDUSTRIAL COPPER MODELLING")
with st.sidebar:
        task = option_menu(menu_title="Navigation", 
                        options= ["Home", "Status Prediction (Classification)", 
                                  "Selling Price Prediction (Regression)"],
                        icons=["house","check2-circle", "currency-dollar"],
                        menu_icon="cast",  
                        default_index=0
                        )
if task == "Home":
    st.subheader("📌 Project Overview")
    st.write(
        """
        Welcome to the **Industrial Copper Modelling App**! 
        "This app predicts the selling price and status based on the User Input"!
        
        This project aims to:
        - Predict the **status** (Won / Lost) of an order using classification.  
        - Predict the **selling price** of copper using regression models.  
        
        You can switch between tasks from the sidebar menu.  
        """
    )
elif task == "Status Prediction (Classification)":
    st.subheader("🔍 Status Prediction")
    st.write("Fill in the required inputs to predict whether the order is *Won* or *Lost*.")
    user_inputs = get_user_inputs(encoders)
    selling_price = st.number_input("Selling Price", min_value=0.0, step=1.0)        
    if st.button("Predict Status"):
        try:
            X_clf = [[
                    user_inputs["quantity_tons"], 
                    user_inputs["thickness"], 
                    user_inputs["width"],
                    user_inputs["item_date_year"],
                    user_inputs["delivery_date_year"],
                    user_inputs["country_enc"],
                    user_inputs["item_type_enc"],
                    user_inputs["application_enc"],
                    user_inputs["product_ref_enc"],
                    user_inputs["material_ref_enc"],
                    selling_price
            ]]         
            y_pred = model_sts.predict(X_clf)[0]
            st.success(f"Predicted Status: {status_map.get(int(y_pred), 'Unknown')}")
            st.balloons()
        except Exception as e:
            st.error(f"Error during prediction: {e}")
elif task == "Selling Price Prediction (Regression)":
    st.subheader("💰 Selling Price Prediction")
    st.write("Fill in the required inputs to predict the **selling price** of copper.")
    user_inputs = get_user_inputs(encoders)
    status = st.selectbox("Status", options=list(status_map.keys()), format_func=lambda x: status_map[x])
    if st.button("Predict Selling Price"):
        try:
            # Build input dataframe (same order as training X for regression)
            X_reg = [[
                    user_inputs["quantity_tons"], 
                    user_inputs["thickness"], 
                    user_inputs["width"],
                    user_inputs["item_date_year"],
                    user_inputs["delivery_date_year"],
                    user_inputs["country_enc"],
                    user_inputs["item_type_enc"],
                    user_inputs["application_enc"],
                    user_inputs["product_ref_enc"],
                    user_inputs["material_ref_enc"],
                    status
            ]]    
            y_pred = model_sp.predict(X_reg)[0]
            st.success(f"Predicted Selling Price: {float(y_pred):.2f}")
            st.balloons()
        except Exception as e:
             st.error(f"Error during prediction: {e}")    
    
    
        




