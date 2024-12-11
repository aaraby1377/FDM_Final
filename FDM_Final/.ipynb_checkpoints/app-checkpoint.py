import streamlit as st
import pickle
import numpy as np

# Load the model and data
pipe = pickle.load(open('rd_model.pkl', 'rb'))
df = pickle.load(open('lapdf.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Mapping for categorical features
company_map = {
    0: 'Acer ', 1: 'Apple ', 2: 'Asus ', 3: 'Chuwi ', 4: 'Dell ',
    5: 'Fujitsu ', 6: 'Google ', 7: 'HP ', 8: 'Huawei ', 9: 'LG ',
    10: 'Lenovo ', 11: 'MSI ', 12: 'Mediacom ', 13: 'Microsoft ',
    14: 'Razer ', 15: 'Samsung ', 16: 'Toshiba ', 17: 'Vero ', 18: 'Xiaomi '
}


typename_map = {
    0: '2 in 1 Convertible ', 1: 'Gaming ', 2: 'Netbook ',
    3: 'Notebook ', 4: 'Ultrabook ', 5: 'Workstation '
}

opsys_map = {
    0: 'Android ', 1: 'Chrome OS ', 2: 'Linux ',
    3: 'Mac OS X ', 4: 'No OS ', 5: 'Windows 10 ',
    6: 'Windows 10 S ', 7: 'Windows 7 ', 8: 'macOS '
}

gpu_brand_map = {
    0: 'AMD', 1: 'ARM', 2: 'Intel',
    3: 'Nvidia'
}

cpu_brand_map = {
    0: 'AMD ',
    1: 'Intel ',
    2: 'Samsung '
}

# Collect user inputs

col1, col2, col3 = st.columns(3)

with col1:
    # Brand
    company = st.selectbox('Company', list(company_map.keys()), format_func=lambda x: company_map[x])

    # Type of laptop
    typename = st.selectbox('Type Name', list(typename_map.keys()), format_func=lambda x: typename_map[x])

    # Inches
    inches = st.selectbox('Inches', df['Inches'].unique())

    # RAM
    ram = st.selectbox('RAM (in GB)', df['Ram'].unique())

    # OS
    opsys = st.selectbox('Operating System', list(opsys_map.keys()), format_func=lambda x: opsys_map[x])

with col2:
    # Weight
    weight = st.selectbox('Weight', df['Weight'].unique())

    # GPU
    gpu = st.selectbox('GPU', list(gpu_brand_map.keys()), format_func=lambda x: gpu_brand_map[x])

    # Screen size
    screen_width = st.selectbox('Screen Width', df['ScreenWidth'].unique())
    screen_height = st.selectbox('Screen Height', df['ScreenHeight'].unique())
    # CPU
    cpu_brand = st.selectbox('CPU Brand', list(cpu_brand_map.keys()), format_func=lambda x: cpu_brand_map[x])

with col3:
    cpu_freq = st.selectbox('CPU Frequency', df['CPU_Frequency'].unique())
    # Storage
    ssd = st.selectbox('SSD (in GB)', df['ssd'].unique())
    flash = st.selectbox('Flash (in GB)', df['flashstorage'].unique())
    hdd = st.selectbox('HDD (in GB)', df['hdd'].unique())
    hybrid = st.selectbox('Hybrid (in GB)', df['hybrid'].unique())


# Predict button
if st.button('Predict Price'):
    # Prepare query
    query = np.array([company, typename, inches, ram, opsys, weight, gpu, screen_width, screen_height, cpu_brand, cpu_freq, ssd, flash, hdd, hybrid])
    query = query.reshape(1, -1)

    # Predict
    predicted_price = pipe.predict(query)[0]

    st.title(f"The predicted price is ${predicted_price:.2f}")
