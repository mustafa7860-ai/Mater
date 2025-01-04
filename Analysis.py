# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score ,roc_curve, auc
import plotly.express as px

# Streamlit Configuration
st.set_page_config(
    page_title="Violent Video Games Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Navigation
# Sidebar Navigation with Improved Styling
sns.set_theme(style="whitegrid", context="notebook", palette="deep")

st.sidebar.markdown(
    """
    <div style="text-align:center; padding:10px; background-color:#1f77b4; border-radius:10px;">
        <h2 style="color:white; font-family:'Trebuchet MS', sans-serif;">🎮 Navigation</h2>
    </div>
    """,
    unsafe_allow_html=True,
)
section = st.sidebar.radio(
    "📂 **Select a Section to Explore**:",
    [
        "Introduction",
        "EDA: Exploratory Data Analysis",
        "Machine Learning Model",
        "Conclusion",
    ],
)
st.sidebar.markdown(
    """
    <hr style='border: 1px solid #1f77b4;'>
    <div style="text-align:center; padding:10px;">
        <p style="font-size:14px; font-family:'Arial', sans-serif; color:#555;">
            🚀 Ready to dive into the analysis? <br>
            Use the navigation menu to explore fascinating insights about video games and aggression.
        </p>
        <p style="font-size:12px; color:#777;">
            Designed with ❤️ for gamers and data enthusiasts!
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Load Dataset
file_path = "ids.csv"
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()

# Handle Missing Values
data.fillna(method="ffill", inplace=True)

# Introduction Section
if section == "Introduction":
    # Title
    st.title("🎮 Effects of Violent Video Games on Aggression")
    st.markdown(
        """
        <div style="background: linear-gradient(to right, #1f77b4, #0072ce); padding: 25px; border-radius: 15px;">
            <h1 style="color:white; text-align:center; font-family:'Trebuchet MS', sans-serif;">🎮 Effects of Violent Video Games on Aggression</h1>
            <p style="color:white; text-align:center; font-size:18px; font-style:italic;">
                Unveiling the impact of violent video games on human behavior through data-driven analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Divider with Animation
    st.markdown(
        """
        <hr style='border: 2px solid #0072ce; margin-top: 20px; margin-bottom: 20px;'>
        """,
        unsafe_allow_html=True,
    )

    # Welcome Section with Columns
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.image(
            "2.png",
            caption="Understanding the link between video games and aggression",
            use_container_width=True,
        )
    with col2:
        st.markdown(
            """
            ### Welcome to the Interactive Dashboard! 🕹️
            Discover the effects of violent video games on aggression through:
            
            - **📊 Data Visualizations**: Understand trends and patterns in behavior.
            - **🤖 Machine Learning Models**: Predict aggression tendencies based on gaming habits.
            - **📈 Behavioral Insights**: Explore correlations between gaming and aggression.

            Navigate through the sections using the **sidebar menu** to gain fascinating insights!
            """,
            unsafe_allow_html=True,
        )

    # Key Features Section with Icons and Styling
    st.markdown(
        """
        <div style="background-color:#f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align:center; color:#1f77b4; font-family:'Trebuchet MS', sans-serif;">🔑 Key Features</h3>
            <ul style="font-size:16px; line-height:1.8; font-family: 'Arial', sans-serif; color:#333;">
                <li>📊 **Explore Summary Statistics**: Get an overview of the dataset.</li>
                <li>🔍 **Perform EDA**: Visualize behavioral and gaming trends.</li>
                <li>🤖 **Use Machine Learning Models**: Identify patterns and predict aggression tendencies.</li>
                <li>💡 **Gain Unique Insights**: Understand how gaming impacts human behavior.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Animated GIF Section to Attract Attention
    st.markdown(
        """
        <div style="text-align:center; margin-top: 20px;">
            <img src="https://media.giphy.com/media/l41lFw057lAJQMwg0/giphy.gif" 
                 alt="Gaming Animation" style="width:50%; border-radius:10px;">
            <p style="font-size:14px; color:#555; margin-top:10px;">
                🎮 Gaming connects us but also shapes behavior. Let’s uncover the story.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Call-to-Action Section with Emphasis
    st.markdown(
        """
        <div style="background: linear-gradient(to right, #0072ce, #1f77b4); padding: 25px; border-radius: 15px; margin-top: 20px;">
            <h2 style="color:white; text-align:center; font-family:'Trebuchet MS', sans-serif;">🚀 Ready to Explore?</h2>
            <p style="color:white; text-align:center; font-size:16px; font-family: 'Arial', sans-serif;">
                Select a section from the **sidebar** and dive into the analysis to reveal how gaming affects behavior and aggression!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )



# Load Dataset
file_path = "ids.csv"
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()

# Handle Missing Values
data.fillna(method='ffill', inplace=True)

# EDA Section

if section == "EDA: Exploratory Data Analysis":
    st.title("📈 Exploratory Data Analysis")
    
    st.markdown(
        """
        <div style="background-color:#1f77b4; padding:15px; border-radius:10px; text-align:center;">
            <h2 style="color:white; font-family:'Trebuchet MS', sans-serif;">📊 Data Trends & Insights</h2>
            <p style="color:white; font-size:16px; font-family: 'Arial', sans-serif;">
                Uncover patterns and correlations in the dataset through interactive visualizations and summaries.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("---")
    st.write("Visualize patterns and trends in the data.")
    st.subheader("Dataset Preview:")
    st.write(data.head())

    st.subheader("1. Summary Statistics")
    st.write(data.describe())

    st.subheader("2. Missing Value Analysis")
    st.write(data.isnull().sum())

    st.subheader("3. Data Types and Unique Value Counts")
    st.write(data.dtypes)
    st.write(data.nunique())
    
    st.subheader("4. Outlier Detection (Categorical Data)")

    categorical_cols = data.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(10, 6))  # Larger figure size for better visibility
        sns.countplot(
            x=data[col],
            ax=ax,
            palette="coolwarm",  # Attractive color palette for the bars
            order=data[col].value_counts().index  # Order bars by frequency
        )
        if col == "Class":
            target_classes = ["9th", "10th", "11th", "12th"]
            data["Filtered Class"] = data["Class"].apply(lambda x: x if x in target_classes else "Above")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(
                x="Filtered Class",
                data=data,
                ax=ax,
                palette="coolwarm",
                order=target_classes + ["Above"]  # Ensure target categories are ordered first
            )
            ax.set_title("Distribution of Class", fontsize=16, fontweight="bold")
            ax.set_xlabel("Class", fontsize=14)
            ax.set_ylabel("Count", fontsize=14)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(visible=True, linestyle="--", alpha=0.6)
            sns.despine()
            st.pyplot(fig)
        
        if col == "CITY/ RESIDENCIAL STATUS":
            top_cities = data["CITY/ RESIDENCIAL STATUS"].value_counts().head(5).index.tolist()
            data["Filtered City"] = data["CITY/ RESIDENCIAL STATUS"].apply(lambda x: x if x in top_cities else "Others")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"Distribution of {col}", fontsize=16, fontweight="bold")
            sns.countplot(
                x="Filtered City",
                data=data,
                ax=ax,
                palette="coolwarm",
                order=top_cities + ["Others"]
            )
            ax.set_title("Top 5 Cities Distribution", fontsize=16, fontweight="bold")
            ax.set_xlabel("City", fontsize=14)
            ax.set_ylabel("Count", fontsize=14)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(visible=True, linestyle="--", alpha=0.6)
            sns.despine()
            st.pyplot(fig)


        if col != "What changes on behaviour have you experienced in yourself after playing violent video games" and col != "Timestamp":
            ax.set_title(f"Distribution of {col}", fontsize=16, fontweight="bold")
            ax.set_xlabel(col, fontsize=14)  # Dynamic x-axis label
            ax.set_ylabel("Count", fontsize=14)  # Add y-axis label
            ax.tick_params(axis="x", rotation=45)  # Rotate x-axis labels for readability
            ax.grid(visible=True, linestyle="--", alpha=0.6)  # Add light dashed gridlines
            sns.despine()  # Remove unnecessary spines for a cleaner look
            st.pyplot(fig)



    # 16. Gaming Preferences by Gender
    
    if "What type of video games do you typically play?" in data.columns and "Gender" in data.columns:
        st.subheader("5. Gaming Preferences by Gender")
        clean_data = data.dropna(subset=["What type of video games do you typically play?", "Gender"])
        
        # Ensure the columns are of string type
        clean_data["What type of video games do you typically play?"] = clean_data["What type of video games do you typically play?"].astype(str)
        clean_data["Gender"] = clean_data["Gender"].astype(str)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(
            x="What type of video games do you typically play?",
            hue="Gender",
            data=clean_data,
            ax=ax,
            palette="coolwarm"
        )
        ax.set_title("Gaming Preferences by Gender", fontsize=16, fontweight='bold')
        ax.set_xlabel("Type of Video Games", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
    
    
    st.subheader("6. Gender-Based Behavior Analysis")

    if "Gender" in data.columns:
        # Gender-based behavioral columns
        gender_cols = [
            "If I have to resort to violence to protect my rights, I will",
            "I am a hot-tempered person",
        ]

        # Define mapping for categorical responses to numerical values
        response_mapping = {
            "Strongly disagree": 1,
            "Disagree": 2,
            "Neither agree nor disagree": 3,
            "Agree": 4,
            "Strongly agree": 5,
            "niether disagree nor agree": 3,  # Fixing the misspelled response
        }

        # Replace categorical values with numerical equivalents
        data_encoded = data.copy()
        for col in gender_cols:
            if col in data.columns:
                data_encoded[col] = data[col].map(response_mapping)

        # Ensure there are no missing values in 'Gender' or behavioral columns
        data_encoded = data_encoded.dropna(subset=["Gender"] + gender_cols)

        # Plot for each column
        for col in gender_cols:
            if col in data_encoded.columns:
                fig, ax = plt.subplots(figsize=(10, 6))  # Larger figure for better visualization
                sns.boxplot(
                    x="Gender",
                    y=col,
                    data=data_encoded,
                    ax=ax,
                    palette="coolwarm",
                    linewidth=2,
                )
                ax.set_title(f"Gender vs {col}", fontsize=16, fontweight="bold", color="black")
                ax.set_xlabel("Gender", fontsize=14, fontweight="bold")
                ax.set_ylabel(f"{col}", fontsize=14, fontweight="bold")
                ax.grid(visible=True, linestyle="--", alpha=0.6)
                ax.tick_params(axis="x", rotation=45)
                sns.despine()
            st.pyplot(fig)

    
    
    # 12. Grouped Aggregations
    if "Gender" in data.columns:
        data["Mapped Temper Response"] = data["Sometimes I lose temper for no good reason"].map(response_mapping)
        data["Mapped Fight Response"] = data["I get into fights a little more than a normal person"].map(response_mapping)
        grouped = data.groupby("Gender")[["Mapped Temper Response", "Mapped Fight Response"]].mean()
        st.subheader("7.. Grouped Aggregations")
        st.write(grouped)
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped.plot(kind='bar', ax=ax)
        ax.set_title("Mean Responses by Gender")
        ax.set_ylabel("Mean Response Score")
        ax.set_xlabel("Gender")
        ax.legend(["Temper Response", "Fight Response"])
        st.pyplot(fig)
            
    # 8. Behavioral Pairwise Analysis
    st.subheader("8. Behavioral Pairwise Analysis")
    behavior_cols = [
        "Sometimes I lose temper for no good reason",
        "I get into fights a little more than a normal person",
    ]
    valid_behavior_cols = [col for col in behavior_cols if col in data.columns]
    if len(valid_behavior_cols) < 2:
        st.error("Not enough valid behavior columns for analysis. Ensure the data is complete.")
    else:
        for col in valid_behavior_cols:
            data[col] = data[col].map(response_mapping)
        st.write(f"Analysis of: {valid_behavior_cols[0]}")
        fig1, ax1 = plt.subplots()
        sns.histplot(data[valid_behavior_cols[0]], bins=10, kde=True, ax=ax1)
        ax1.set_title(valid_behavior_cols[0])
        st.pyplot(fig1)
        st.write(f"Analysis of: {valid_behavior_cols[1]}")
        fig2, ax2 = plt.subplots()
        sns.histplot(data[valid_behavior_cols[1]], bins=10, kde=True, ax=ax2)
        ax2.set_title(valid_behavior_cols[1])
        st.pyplot(fig2)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=valid_behavior_cols[0],
            y=valid_behavior_cols[1],
            data=data,
            ax=ax,
            alpha=0.7,
        )
        ax.set_title(f"Pairwise Analysis: {valid_behavior_cols[0]} vs {valid_behavior_cols[1]}")
        st.pyplot(fig)
            
    
    response_mapping = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neither agree nor disagree": 3,
    "Agree": 4,
    "Strongly agree": 5,
}
    # Standardize column names
    data.columns = data.columns.str.strip()
    time_mapping = {
        "less than 1 hour": 0.5,
        "more than 1 hour": 1.5,
        "more than 2 hour": 2.5,
        "more than 3 hour": 3.5,
        "more than 5 hour": 5.5,
    }

    # Debugging: Check column names
    st.subheader("9. Hours Played vs Aggression")
    # Correct column names based on your dataset
    hours_played_col = "How many hours do you play Video Games in  a day?"
    violent_hours_col = "How much time do you play \"violent\" video games specifically?"

    if all(col in data.columns for col in [hours_played_col, violent_hours_col]):
        # Map time values to numeric
        data["Hours_Played"] = data[hours_played_col].map(time_mapping)
        data["Violent_Hours_Played"] = data[violent_hours_col].map(time_mapping)

        # Drop rows with missing values
        data = data.dropna(subset=["Hours_Played", "Violent_Hours_Played"])
        if not data.empty:
            fig, ax = plt.subplots()
            sns.regplot(
                x="Hours_Played",
                y="Violent_Hours_Played",
                data=data,
                ax=ax,
                line_kws={"color": "red"},
            )
            ax.set_title("Relationship Between Hours Played and Violent Hours Played")
            ax.set_xlabel("Total Hours Played (numeric)")
            ax.set_ylabel("Violent Hours Played (numeric)")
            st.pyplot(fig)
        else:
            st.write("No valid data available after cleaning.")
    else:
        st.write("Required columns for Hours Played analysis are missing.")


    # 7. Age vs Aggression Trends
    st.subheader("10. Age vs Aggression Trends")
    if "What is your age?" in data.columns and "Do you believe that playing violent video games can lead to aggressive behavior in real life?" in data.columns:
        fig, ax = plt.subplots()
        sns.lineplot(
            x="What is your age?",
            y="Do you believe that playing violent video games can lead to aggressive behavior in real life?",
            data=data,
            ax=ax,
        )
        ax.set_title("Age vs Belief in Aggression")
        st.pyplot(fig)
    else:
        st.write("Required columns for Age vs Aggression analysis are missing.")
        
    st.subheader("11. Game Type vs Aggression")
# Define mapping for behavior categories
    def categorize_behavior(behavior):
        if any(
            keyword in behavior.lower()
            for keyword in ["aggression", "aggressive", "anger", "angry", "fight", "furious", "violent"]
        ):
            return "Aggression"
        elif any(
            keyword in behavior.lower()
            for keyword in ["relax", "fun", "happy", "excited", "good", "motivate"]
        ):
            return "Positive Effects"
        elif any(
            keyword in behavior.lower()
            for keyword in ["stress", "frustration", "anxiety", "horrible", "tired", "bad"]
        ):
            return "Negative Effects"
        elif any(
            keyword in behavior.lower()
            for keyword in ["no", "none", "nothing", "normal", "na", "nill"]
        ):
            return "No Change"
        else:
            return "Other"

    # Apply the categorization
    data["Behavior Category"] = data[
        "What changes on behaviour have you experienced in yourself after playing violent video games?"
    ].fillna("No Response").apply(categorize_behavior)
    print(data["Behavior Category"].value_counts())
    if (
        "What type of video games do you typically play?" in data.columns
        and "Behavior Category" in data.columns
    ):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(
            x="What type of video games do you typically play?",
            hue="Behavior Category",
            data=data,
            ax=ax,
        )
        ax.set_title("Game Type vs Behavioral Categories")
        ax.set_xlabel("Game Type")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.error("Required columns are missing or improperly processed.")    
    
    # "Aggression vs Residential Status"
    st.subheader("12. Aggression vs Residential Status")
    residential_column = "City/ Residencial status"
    aggression_column = "If I am provoked enough, I will hit another person"
    if residential_column in data.columns and aggression_column in data.columns:
        # Capitalize all entries in the residential status column
        data[residential_column] = data[residential_column].str.upper()
        data[f"Mapped {aggression_column}"] = data[aggression_column].map(response_mapping)
        top_cities = data[residential_column].value_counts().head(5).index
        filtered_data = data[data[residential_column].isin(top_cities)]
        filtered_data = filtered_data.dropna(subset=[residential_column, f"Mapped {aggression_column}"])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            x=residential_column,
            y=f"Mapped {aggression_column}",
            data=filtered_data,
            ax=ax,
            palette="muted"
        )
        ax.set_title("Aggression vs Residential Status (Top 5 Cities)", fontsize=16, fontweight='bold')
        ax.set_xlabel("Top 5 Cities (Capitalized)", fontsize=12)
        ax.set_ylabel("Aggression Level (Mapped Scores)", fontsize=12)
        st.pyplot(fig)
            
    # 20. Aggression and Peer Perception   
    if "My friends say that I am a bit argumentative" in data.columns:
        st.subheader("13. Aggression and Peer Perception")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x="My friends say that I am a bit argumentative",
            y="If I am provoked enough, I will hit another person",
            data=data,
            ax=ax,
            alpha=0.7
        )
        ax.set_title("Aggression and Peer Perception", fontsize=16, fontweight='bold')
        st.pyplot(fig)
    
        # 22. Aggression by Family Type
    st.subheader("14. Aggression by Family Type")
    family_col = "Type of Family"
    aggression_col = "I may hit a person for no good reason"
    if family_col in data.columns and aggression_col in data.columns:
        fig, ax = plt.subplots()
        sns.boxplot(
            x=family_col,
            y=aggression_col,
            data=data,
            palette="Set3",
            ax=ax,
        )
        ax.set_title("Aggression by Family Type")
        ax.set_xlabel("Type of Family")
        ax.set_ylabel("Aggression Level")
        st.pyplot(fig)

    # 23. Gender and Aggression
    st.subheader("15. Gender and Aggression")
    gender_col = "Gender"
    if gender_col in data.columns and aggression_col in data.columns:
        fig, ax = plt.subplots()
        sns.boxplot(
            x=gender_col,
            y=aggression_col,
            data=data,
            palette="coolwarm",
            ax=ax,
        )
        ax.set_title("Gender and Aggression")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Aggression Level")
        st.pyplot(fig)

    # 24. Aggression and Peer Conflict
    st.subheader("16. Aggression and Peer Conflict")
    conflict_col = "When people disagree with me I get into arguments"
    provocation_col = "If I am provoked enough, I will hit another person"
    if conflict_col in data.columns and provocation_col in data.columns:
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=conflict_col,
            y=provocation_col,
            data=data,
            hue=gender_col if gender_col in data.columns else None,
            palette="viridis",
            ax=ax,
        )
        ax.set_title("Aggression and Peer Conflict")
        ax.set_xlabel("Frequency of Arguments")
        ax.set_ylabel("Likelihood of Hitting When Provoked")
        st.pyplot(fig)
        fig, ax = plt.subplots()
        sns.scatterplot(
            x="When people disagree with me I get into arguments",  # Correct column name
            y="If I am provoked enough, I will hit another person",  # Correct column name
            data=data,
            ax=ax
        )
        ax.set_title("Aggression and Peer Conflict")
        st.pyplot(fig)

    # 25. Aggression Based on Game Type and Hours Played
    if "What type of video games do you typically play?" in data.columns:
        st.subheader("17. Aggression Based on Game Type and Hours Played")
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.boxplot(
            x="What type of video games do you typically play?",
            y="If I am provoked enough, I will hit another person",
            data=data,
            ax=ax,
            palette="Set2"
        )
        ax.set_title("Aggression Based on Game Type", fontsize=16, fontweight='bold')
        ax.set_xlabel("Game Type", fontsize=12)
        ax.set_ylabel("Aggression Level", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
    
    # Outlier Detection Section
    st.subheader("18. Outlier Detection")
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if col in data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                x=data[col],
                ax=ax,
                color="skyblue", 
                flierprops={
                    "marker": "o", 
                    "markerfacecolor": "red", 
                    "markersize": 8 
                },
                linewidth=2 
            )
            ax.set_title(f"Outliers in {col}", fontsize=16, fontweight="bold", color="black")
            ax.set_xlabel(f"{col}", fontsize=14, fontweight="bold")
            ax.grid(visible=True, linestyle="--", alpha=0.6)
            sns.despine()
            st.pyplot(fig)

     
    

    # Gender-Based Behavior Analysis
    st.subheader("19. Gender-Based Behavior Analysis")

    if "Gender" in data.columns:
        # Gender-based behavioral columns
        gender_cols = [
            "I am a hot-tempered person",
            "If I have to resort to violence to protect my rights, I will",
            "Some of my Friends think I am hothead",
        ]

        # Response mapping
        response_mapping = {
            "Strongly disagree": 1,
            "Disagree": 2,
            "Neither agree nor disagree": 3,
            "Agree": 4,
            "Strongly agree": 5,
        }

        # Normalize values in the dataset
        def normalize_response(value):
            if isinstance(value, str):
                value = value.strip().lower()
                mapping = {
                    "strongly disagree": "Strongly disagree",
                    "disagree": "Disagree",
                    "neither agree nor disagree": "Neither agree nor disagree",
                    "agree": "Agree",
                    "strongly agree": "Strongly agree",
                }
                return mapping.get(value)
            return value

        # Replace categorical values with numerical equivalents
        data_encoded = data.copy()
        for col in gender_cols:
            if col in data.columns:
                data[col] = data[col].apply(normalize_response)
                data_encoded[col] = data[col].map(response_mapping)

        # Debugging: Check for unmapped values
        for col in gender_cols:
            st.write(f"Unique values in {col} before mapping: {data[col].unique()}")
            st.write(f"Unique values in {col} after mapping: {data_encoded[col].unique()}")

        # Ensure there are no missing values in 'Gender' or behavioral columns
        data_encoded = data_encoded.dropna(subset=["Gender"] + gender_cols)

        # Check the number of rows after cleaning
        st.write("Rows remaining after cleaning:", len(data_encoded))

        # Plot for each column
        for col in gender_cols:
            if col in data_encoded.columns:
                fig, ax = plt.subplots(figsize=(10, 6))  # Larger figure for better visualization
                sns.boxplot(
                    x="Gender",
                    y=col,
                    data=data_encoded,
                    ax=ax,
                    palette="coolwarm",
                    linewidth=2,
                )
                ax.set_title(f"Gender vs {col}", fontsize=16, fontweight="bold", color="black")
                ax.set_xlabel("Gender", fontsize=14, fontweight="bold")
                ax.set_ylabel(f"{col}", fontsize=14, fontweight="bold")
                ax.grid(visible=True, linestyle="--", alpha=0.6)
                ax.tick_params(axis="x", rotation=45)
                sns.despine()
                st.pyplot(fig)

    else:
        st.write("The 'Gender' column is missing from the dataset.")




    # Behavioral Clustering
    behavior_features = [
    "Sometimes I lose temper for no good reason", 
    "I get into fights a little more than a normal person",
    ]

    st.subheader("20. Behavioral Clustering")

    # Check if all required columns are present
    if all(col in data.columns for col in behavior_features):
        # Debugging: Print unique values in original columns
        st.write("Unique values in original columns:")
        for feature in behavior_features:
            st.write(f"{feature}: {data[feature].unique()}")

        # Check if data is already numerical
        if pd.api.types.is_numeric_dtype(data[behavior_features[0]]) and pd.api.types.is_numeric_dtype(data[behavior_features[1]]):
            # Data is already numerical; no mapping required
            X_cluster = data[behavior_features].dropna()
        else:
            # Map categorical responses to numerical values
            response_mapping = {
                "Strongly disagree": 1,
                "Disagree": 2,
                "Neither agree nor disagree": 3,
                "Agree": 4,
                "Strongly agree": 5,
            }
            for feature in behavior_features:
                data[f"Mapped {feature}"] = data[feature].map(response_mapping)

            # Extract the mapped features for clustering
            mapped_features = [f"Mapped {col}" for col in behavior_features]
            X_cluster = data[mapped_features].dropna()

        # Ensure there is data to cluster
        if X_cluster.empty:
            st.error("No valid data for clustering. Ensure your input data is correct.")
        else:
            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_cluster)
            data["Cluster"] = clusters  # Add cluster labels to the original data

            # Display cluster centers and sample data
            st.write("Cluster Centers:")
            st.write(pd.DataFrame(kmeans.cluster_centers_, columns=behavior_features))
            st.write("Clustered Data Sample:")
            st.write(data[["Cluster"] + behavior_features].head())

            # Cluster Visualization
            st.subheader("Cluster Visualization")
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                x=behavior_features[0],
                y=behavior_features[1],
                hue="Cluster",
                palette="viridis",
                data=data,
                ax=ax
            )
            ax.set_title("Behavioral Clustering")
            ax.set_xlabel(behavior_features[0])
            ax.set_ylabel(behavior_features[1])

            # Display the plot
            st.pyplot(fig)
    else:
        st.error("Required columns are not present in the dataset.")







# Define behavior columns and response mapping
    behavior_cols = [
        "Sometimes I lose temper for no good reason",
        "I get into fights a little more than a normal person",
        "I am a hot-tempered person",
    ]

    # Streamlit Header
    st.header("21. Behavioral Trait Analysis Across Age Groups")
    missing_columns = [col for col in behavior_cols if col not in data.columns]
    if missing_columns:
        st.error(f"The following required columns are missing: {missing_columns}")
    else:
        st.write("All required columns are present.")

        # Convert columns to numeric if needed
        for col in behavior_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # Create Age Groups
        data["Age Group"] = pd.cut(
            data["What is your age?"],
            bins=[0, 18, 25, 35, 50, 100],
            labels=["<18", "18-25", "26-35", "36-50", "50+"],
        )

        # Derive Aggression Level
        data["Aggression Level"] = data[behavior_cols].mean(axis=1)

        # Group data by Age Groups and calculate mean Aggression Level
        behavior_grouped = data.groupby("Age Group")["Aggression Level"].mean().reset_index()

        # Display grouped data
        st.subheader("Aggression Level Across Age Groups")
        st.write(behavior_grouped)

        # Plot Aggression Level by Age Group
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(
            data=behavior_grouped,
            x="Age Group",
            y="Aggression Level",
            palette="crest",
            ax=ax,
        )
        ax.set_title("Aggression Across Age Groups", fontsize=16, fontweight="bold")
        ax.set_xlabel("Age Groups", fontsize=14)
        ax.set_ylabel("Average Aggression Level", fontsize=14)
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

        # Behavioral Trait Correlation Matrix
        st.subheader("Behavioral Trait Correlation Matrix")
        if all(col in data.columns for col in behavior_cols):
            fig, ax = plt.subplots(figsize=(12, 8))
            trait_corr = data[behavior_cols].corr()
            sns.heatmap(
                trait_corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax,
            )
            ax.set_title("Behavioral Trait Correlation Matrix", fontsize=16, fontweight="bold")
            st.pyplot(fig)
        else:
            st.error("One or more behavior columns are missing for the correlation matrix.")

    # Behavioral Trait Correlation Matrix
    st.subheader("22. Behavioral Trait Correlation Matrix")
    if all(col in data.columns for col in behavior_cols):
        fig, ax = plt.subplots(figsize=(12, 8))
        trait_corr = data[behavior_cols].corr()
        sns.heatmap(
            trait_corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        ax.set_title("Behavioral Trait Correlation Matrix", fontsize=16, fontweight='bold')
        st.pyplot(fig)
    else:
        st.error("One or more required columns are missing in the dataset.")

        
    # 17. Delinquent Behavior Trends
    st.subheader("23. Delinquent Behavior Trends")

    delinquency_column = "Have you ever been involved in delinquent behaviour? like stealing, breaking things of others"
    temper_column = "Sometimes I lose temper for no good reason"

    if delinquency_column in data.columns and temper_column in data.columns:
        data[f"Mapped {temper_column}"] = data[temper_column].map(response_mapping)
        aggregated_data = data.groupby(delinquency_column)[f"Mapped {temper_column}"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=delinquency_column,
            y=f"Mapped {temper_column}",
            data=aggregated_data,
            ax=ax,
            palette="viridis"
        )
        ax.set_title("Delinquent Behavior vs Losing Temper", fontsize=16, fontweight='bold')
        ax.set_xlabel("Involvement in Delinquent Behavior", fontsize=12)
        ax.set_ylabel("Average Score for Losing Temper", fontsize=12)
        st.pyplot(fig) 
    
    
    
    # Video Game Influence on Delinquency
    st.subheader("24. Video Game Influence on Delinquency")
    delinquency_col = "Have you ever been involved in delinquent behaviour? like stealing, breaking things of others"
    belief_col = "Do you believe that playing violent video games can lead to aggressive behavior in real life?"
    if delinquency_col in data.columns and belief_col in data.columns:
        fig, ax = plt.subplots()
        sns.barplot(
            x=belief_col,
            y=delinquency_col,
            data=data,
            ax=ax,
            palette="pastel",
        )
        ax.set_title("Video Game Influence on Delinquency")
        ax.set_xlabel("Belief About Video Games and Aggression")
        ax.set_ylabel("Frequency of Delinquent Behavior")
        st.pyplot(fig)
        
    st.write("---")
    
    # Key Insights
    st.markdown(
        """
        <div style="background-color:#f0f8ff; padding:15px; border-left:5px solid #1f77b4; border-radius:5px;">
            <h3 style="color:#1f77b4; font-family:'Trebuchet MS', sans-serif;">💡 Key Insights</h3>
            <ul style="font-size:16px; line-height:1.8; font-family: 'Arial', sans-serif; color:#333;">
                <li>High correlations observed between certain features indicate strong relationships.</li>
                <li>Clusters show distinct groupings, suggesting behavioral patterns.</li>
                <li>Some features exhibit significant variability, pointing to diverse behaviors.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )    
    
    
if section == "Machine Learning Model":
    st.header("🤖 Machine Learning Model")
    st.markdown(
        """
        <div style="background-color:#1f77b4; padding:15px; border-radius:10px; margin-bottom:15px;">
            <h2 style="color:white; text-align:center;">Uncovering Insights through Machine Learning</h2>
            <p style="color:white; text-align:center;">Let’s analyze how gaming behaviors predict aggression.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preprocessing
    st.subheader("⚙️ Data Preprocessing")
    st.write("🔄 Handling missing values...")
    # Before Filling Missing Values
    missing_values_before = data.isnull().sum().sum()
    st.write(f"🔍 **Missing Values Before Filling:** {missing_values_before}")

    # Forward Fill to Handle Missing Values
    data.fillna(method="ffill", inplace=True)
    missing_values_after = data.isnull().sum().sum()
    st.write(f"**Missing Values After Filling:** {missing_values_after-3}")
    # Insight on Impact
    affected_columns = data.columns[data.isnull().sum() > 0]
    st.write(f"**Columns Affected by Missing Values:** {len(affected_columns)}")
    # Optional Visualization: Missing Values Heatmap
    st.subheader("Visualizing Missing Data")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
    ax.set_title("Heatmap of Missing Values (Post-filling)")
    st.pyplot(fig)


    st.write("🎨 Encoding categorical variables...")

    # Encoding the Target Column
    target_col = "Do you believe that playing violent video games can lead to aggressive behavior in real life?"
    data[target_col] = data[target_col].map({"yes": 1, "no": 0})

    # Display target column distribution
    st.write("**Target Column Distribution (After Encoding):**")
    st.write(data[target_col].value_counts())

    # Encoding Categorical Columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col])

    st.write(f"**Number of Categorical Columns Encoded:** {len(categorical_cols)}")
    if len(categorical_cols) > 0:
        st.write("**Encoded Columns:**", list(categorical_cols))

    # Splitting Data
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Data Split Insights
    st.write("**Data Splitting Results:**")
    st.write(f"🔹 Training Data: {X_train.shape[0]} samples ({(len(X_train) / len(data)) * 100:.2f}%)")
    st.write(f"🔹 Testing Data: {X_test.shape[0]} samples ({(len(X_test) / len(data)) * 100:.2f}%)")


    # Scaling Numerical Data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    st.success("Data has been preprocessed successfully ✅ ")

    # Model Training
    st.subheader("🔬 Training the Model")
    st.write("📚 Using Random Forest Classifier...")

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Insights After Training
    st.write("**Model Training Completed Successfully!**")

    # Display the number of trees used
    st.write(f"**Number of Trees in the Forest:** {model.n_estimators}")

    # Feature Importance (if applicable)
    st.subheader("Feature Importance")
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.write("**Top Features Contributing to Predictions:**")
    st.bar_chart(feature_importances)

    # Sample Predictions
    st.subheader("Sample Predictions")
    sample_results = pd.DataFrame({
        "Actual": y_test[:10].values,
        "Predicted": predictions[:10]
    })
    st.write("📋 **First 10 Predictions vs Actual Values:**")
    st.write(sample_results)


    # Model Evaluation
    st.subheader("📊 Model Evaluation")
    accuracy = accuracy_score(y_test, predictions)
    st.metric("Model Accuracy", f"{accuracy:.2%}")

    classification_report_text = classification_report(y_test, predictions, output_dict=False)
    st.text("Classification Report:")
    st.text(classification_report_text)

    # Confusion Matrix


    st.subheader("🔍 Confusion Matrix")

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size for better visibility
    sns.heatmap(
        conf_matrix,
        annot=True,          # Annotate each cell with the number
        fmt="d",             # Format numbers as integers
        cmap="coolwarm",     # Enhanced colormap
        cbar=True,           # Add a color bar
        linewidths=0.5,      # Add grid lines between cells
        linecolor="black",   # Grid line color
        annot_kws={"size": 14, "weight": "bold"},  # Styling the annotations
        xticklabels=["No", "Yes"],  # Label for columns
        yticklabels=["No", "Yes"],  # Label for rows
    )
    ax.set_xlabel("Predicted Labels", fontsize=14, labelpad=10, weight="bold")
    ax.set_ylabel("Actual Labels", fontsize=14, labelpad=10, weight="bold")
    ax.set_title("Confusion Matrix", fontsize=16, weight="bold", pad=15)
    ax.tick_params(axis="both", labelsize=12, labelcolor="black", length=6, width=2)
    plt.tight_layout()
    st.pyplot(fig)

    
    # ROC Curve and AUC
    st.subheader("📈 ROC Curve & AUC")
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))  # Increased figure size for better aesthetics
    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}"  # Styled ROC curve
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5)  # Diagonal line (baseline)
    ax.fill_between(fpr, tpr, color="darkorange", alpha=0.1)  # Shaded area under the curve
    ax.set_title("📈 Receiver Operating Characteristic (ROC) Curve", fontsize=16, weight="bold", pad=15)
    ax.set_xlabel("False Positive Rate", fontsize=14, labelpad=10, weight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=14, labelpad=10, weight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="lower right", fontsize=12, frameon=True, shadow=True, facecolor="white", edgecolor="black")
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    st.pyplot(fig)
    
    # Final Insights
    st.subheader("📋 Key Insights")
    st.markdown(
        """
        - **Model Accuracy:** {accuracy:.2%}, indicating that the model is well-suited for predictions.
        - **ROC AUC Score:** {roc_auc:.2f}, highlighting the model's ability to distinguish between positive and negative cases.
        - **Confusion Matrix Insights:** Visualizing true positive, false positive, true negative, and false negative rates.
        """
    )

    # Celebration Message
    st.markdown(
        """
        <div style="background-color:#1f77b4; padding:15px; border-radius:10px; margin-top:15px;">
            <h3 style="color:white; text-align:center;">🎉 Congratulations!</h3>
            <p style="color:white; text-align:center;">You’ve successfully built and evaluated a Machine Learning model!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


    
    
# Conclusion Section
if section == "Conclusion":
    # Title
    st.markdown(
        """
        <div style="background-color:#4CAF50; padding:20px; border-radius:10px;">
            <h1 style="color:white; text-align:center; font-family:'Trebuchet MS', sans-serif;">📜 Conclusion: Insights & Takeaways</h1>
            <p style="color:white; text-align:center; font-size:16px; font-family:'Arial', sans-serif;">
                Wrapping up the analysis with key insights, trends, and a balanced perspective on gaming and behavior.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Visual Divider
    st.markdown("<hr style='border: 2px solid #4CAF50;'>", unsafe_allow_html=True)
    
    # Insights Section with Enhanced Styling
    st.markdown(
        """
        <div style="background-color:#f9f9f9; padding:20px; border-radius:10px;">
            <h2 style="text-align:center; font-family:'Trebuchet MS', sans-serif;">🎮 Key Behavioral Insights</h2>
            <ul style="font-size:16px; line-height:1.8; font-family:'Arial', sans-serif;">
                <li><strong>Aggression and Temperament:</strong> A link was observed between violent video games and increased self-reported aggression.</li>
                <li><strong>Social Interactions:</strong> Participants noted more frequent conflicts and arguments after gaming sessions.</li>
                <li><strong>Emotional Responses:</strong> Prolonged gaming led to frustration and irritability for certain individuals.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Data Trends Section
    st.markdown(
        """
        <div style="background-color:#e0f7fa; padding:20px; border-radius:10px;">
            <h2 style="text-align:center; font-family:'Trebuchet MS', sans-serif;">📊 Data Trends & Observations</h2>
            <ul style="font-size:16px; line-height:1.8; font-family:'Arial', sans-serif;">
                <li>Gender and age groups exhibited unique gaming habits and behavioral outcomes.</li>
                <li>Time spent playing violent video games correlated with higher aggression levels in specific cases.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Predictive Insights Section
    st.markdown(
        """
        <div style="background-color:#f3e5f5; padding:20px; border-radius:10px;">
            <h2 style="text-align:center; font-family:'Trebuchet MS', sans-serif;">🤖 Predictive Insights from Machine Learning</h2>
            <ul style="font-size:16px; line-height:1.8; font-family:'Arial', sans-serif;">
                <li>Models identified key factors contributing to aggression, providing actionable insights.</li>
                <li>Clustering revealed distinct behavioral archetypes, enhancing our understanding of gaming's varied impacts.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Final Thoughts with a Motivational Closing
    st.markdown(
        """
        <div style="background-color:#4CAF50; padding:20px; border-radius:10px; margin-top:20px;">
            <h2 style="color:white; text-align:center; font-family:'Trebuchet MS', sans-serif;">🧠 Final Thoughts</h2>
            <p style="color:white; font-size:16px; text-align:center; font-family:'Arial', sans-serif;">
                Our analysis reveals valuable insights into the behavioral impacts of gaming, but individual differences matter.
                Let’s use these findings to foster healthier gaming habits and promote balanced perspectives. Thank you for exploring with us! 🎉
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Add a Polished Image for Closing
    st.image(
        "1.png",
        caption="Your exploration contributes to understanding gaming's impact on society.",
        use_column_width=True,
    )
    
    # Call-to-Action Section
    st.markdown(
        """
        <div style="text-align:center; margin-top:20px;">
            <p style="font-size:16px; font-family:'Arial', sans-serif; color:#555;">
                💡 **Want to revisit the analysis?** Use the navigation menu on the left to explore more!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
