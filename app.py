import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Customer Analytics",
    page_icon="🛒",
    layout="wide"
)

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
clv_model = pickle.load(open("clv_model.pkl", "rb"))
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("cleaned_online_retail.csv")

# Create TotalAmount if not already present
if "TotalAmount" not in df.columns:
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# Convert InvoiceDate
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# -------------------------------------------------
# CREATE RFM TABLE
# -------------------------------------------------
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

customer_df = (
    df.groupby("CustomerID")
    .agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalAmount": "sum"
    })
    .reset_index()
)

customer_df.columns = [
    "CustomerID",
    "Recency",
    "Frequency",
    "Monetary"
]

# -------------------------------------------------
# CUSTOMER SEGMENTS
# -------------------------------------------------
scaled = scaler.transform(
    customer_df[["Recency", "Frequency", "Monetary"]]
)

customer_df["Cluster"] = kmeans.predict(scaled)

segment_names = {
    0: "VIP Customer",
    1: "Loyal Customer",
    2: "Potential Customer",
    3: "At Risk Customer"
}

customer_df["Segment"] = customer_df["Cluster"].map(segment_names)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("🛒 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Dashboard",
        "👥 Customer Segmentation",
        "💰 CLV Prediction",
        "📈 Model Performance",
        "ℹ️ About"
    ]
)

# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title("🛒 E-Commerce Customer Lifetime Value & Segmentation")

    st.markdown("---")

    st.subheader("📈 Business Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customers",
        f"{df['CustomerID'].nunique():,}"
    )

    col2.metric(
        "Orders",
        f"{df['InvoiceNo'].nunique():,}"
    )

    col3.metric(
        "Revenue",
        f"£{df['TotalAmount'].sum():,.2f}"
    )

    col4.metric(
        "Countries",
        f"{df['Country'].nunique()}"
    )

    st.markdown("---")

    st.write("""
This project predicts **Customer Lifetime Value (CLV)** and performs **Customer Segmentation**
using Machine Learning.

### Machine Learning Models
- K-Means Clustering
- Random Forest Regression

### Technologies Used
- Python
- Pandas
- Scikit-learn
- Plotly
- Streamlit

### Internship
Data Science & AI Summer School 2026
""")

# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Dashboard":

    st.title("📊 Business Dashboard")

    st.subheader("Top 10 Countries by Revenue")

    country = (
        df.groupby("Country")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig1 = px.bar(
        country,
        x="Country",
        y="TotalAmount",
        title="Top Countries by Revenue"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Top Selling Products")

    products = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig2 = px.bar(
        products,
        x="Quantity",
        y="Description",
        orientation="h",
        title="Top Selling Products"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Customer Segment Distribution")

    segment_counts = (
        customer_df["Segment"]
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = [
        "Segment",
        "Customers"
    ]

    fig3 = px.pie(
        segment_counts,
        names="Segment",
        values="Customers",
        title="Customer Segments"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif page == "👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")

    recency = st.number_input(
        "Recency (Days)",
        min_value=0,
        value=30
    )

    frequency = st.number_input(
        "Frequency",
        min_value=1,
        value=5
    )

    monetary = st.number_input(
        "Monetary (£)",
        min_value=0.0,
        value=500.0
    )

    if st.button("Predict Segment"):

        sample = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=[
                "Recency",
                "Frequency",
                "Monetary"
            ]
        )

        sample_scaled = scaler.transform(sample)

        cluster = kmeans.predict(sample_scaled)[0]

        segment = segment_names.get(cluster, "Unknown")

        st.success(f"Predicted Segment: {segment}")

        if cluster == 0:

            st.info("""
### Recommendation

- Premium Membership
- Exclusive Discounts
- Early Product Access
- Personal Shopping Experience
""")

        elif cluster == 1:

            st.info("""
### Recommendation

- Loyalty Rewards
- Bundle Offers
- Referral Coupons
""")

        elif cluster == 2:

            st.info("""
### Recommendation

- Welcome Coupons
- Email Campaigns
- Product Recommendations
""")

        else:

            st.info("""
### Recommendation

- Win-back Offers
- Reminder Emails
- Personalised Promotions
""")
# ============================================================
# CLV PREDICTION
# ============================================================

elif page == "💰 CLV Prediction":

    st.title("💰 Customer Lifetime Value Prediction")

    st.write("Enter customer details below to predict the estimated Customer Lifetime Value (CLV).")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        recency = st.number_input(
            "Recency (Days)",
            min_value=0,
            value=30,
            key="clv_recency"
        )

        frequency = st.number_input(
            "Frequency",
            min_value=1,
            value=5,
            key="clv_frequency"
        )

    with col2:
        monetary = st.number_input(
            "Monetary (£)",
            min_value=0.0,
            value=500.0,
            key="clv_monetary"
        )

    if st.button("Predict CLV"):

        input_df = pd.DataFrame({
            "Recency": [recency],
            "Frequency": [frequency],
            "Monetary": [monetary]
        })

        prediction = clv_model.predict(input_df)[0]

        st.success(f"Predicted Customer Lifetime Value: £{prediction:,.2f}")

        st.markdown("---")

        if prediction >= 5000:

            customer_type = "🌟 High Value Customer"

            recommendation = """
- Premium Membership
- VIP Support
- Exclusive Rewards
- Early Product Access
"""

        elif prediction >= 2000:

            customer_type = "⭐ Medium Value Customer"

            recommendation = """
- Loyalty Programme
- Bundle Offers
- Seasonal Discounts
- Personalised Emails
"""

        else:

            customer_type = "🌱 Low Value Customer"

            recommendation = """
- Welcome Coupon
- Email Marketing
- Product Recommendations
- Discount Campaigns
"""

        st.subheader(customer_type)

        st.markdown("### Recommended Business Strategy")

        st.markdown(recommendation)

        result = pd.DataFrame({
            "Recency": [recency],
            "Frequency": [frequency],
            "Monetary": [monetary],
            "Predicted_CLV": [round(prediction, 2)]
        })

        st.download_button(
            label="📥 Download Prediction",
            data=result.to_csv(index=False),
            file_name="clv_prediction.csv",
            mime="text/csv"
        )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    st.write("Performance metrics of the Machine Learning models.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Silhouette Score", "0.616")
    col2.metric("RMSE", "541.47")
    col3.metric("MAE", "35.97")

    col4, col5 = st.columns(2)

    col4.metric("MAPE", "0.21%")
    col5.metric("R² Score", "0.9971")

    st.markdown("---")

    metrics = pd.DataFrame({
        "Metric": [
            "Silhouette Score",
            "RMSE",
            "MAE",
            "MAPE",
            "R² Score"
        ],
        "Value": [
            "0.616",
            "541.47",
            "35.97",
            "0.21%",
            "0.9971"
        ]
    })

    st.dataframe(metrics, use_container_width=True)

# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
## E-Commerce Customer Lifetime Value (CLV) & Segmentation

### Objective

This project helps businesses identify valuable customers and estimate their future Customer Lifetime Value using Machine Learning.

---

### Machine Learning Models

- K-Means Clustering
- Random Forest Regression

---

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Streamlit

---

### Features

- Customer Segmentation
- CLV Prediction
- Interactive Dashboard
- Business Recommendations
- Download Prediction Report

---

### Internship

**Data Science & AI Summer School 2026**

---

### Developed By

**Bidyamanjari Jena**
""")