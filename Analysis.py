import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans 

# Streamlit Application
st.title("Effects of Violent Video Games on Aggression")
st.write("This application performs an end-to-end analysis and machine learning modeling on the dataset.")
sns.set_theme(style="whitegrid", context="notebook", palette="deep")

# Load Dataset
st.header("Dataset")
file_path = "ids.csv"
data = pd.read_csv(file_path)

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# st.write("### Dataset Preview:")
# st.write(data.head())

# # 1. Summary Statistics
# st.subheader("1. Summary Statistics")
# st.write(data.describe())

# # 2. Missing Values
# st.subheader("2. Missing Values")
# st.write(data.isnull().sum())

# # 3. Correlation Heatmap
# st.subheader("3. Correlation Heatmap")
# if not data.select_dtypes(include=["float64", "int64"]).empty:
#     corr = data.corr(numeric_only=True)
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
#     st.pyplot(fig)
# else:
#     st.write("No numerical data available for correlation analysis.")

# # 4. Feature Distributions
# st.subheader("4. Feature Distributions")
# if "What is your age?" in data.columns:
#     # Convert to numeric if necessary
#     data["What is your age?"] = pd.to_numeric(data["What is your age?"], errors="coerce")
#     fig, ax = plt.subplots()
#     sns.histplot(data["What is your age?"], kde=True, bins=30, ax=ax)
#     ax.set_title("Distribution of Age")
#     st.pyplot(fig)
# else:
#     st.write("The 'What is your age?' column is missing or not numeric.")

# # 5. Gender-Based Behavior Analysis
# st.subheader("5. Gender-Based Behavior Analysis")
# if "Gender" in data.columns:
#     gender_cols = [
#         "Some of my Friends think I am hothead",
#         "If I have to resort to violence to protect my rights, I will",
#         "I am a hot-tempered person",
#     ]
#     for col in gender_cols:
#         if col in data.columns:
#             fig, ax = plt.subplots()
#             sns.boxplot(x="Gender", y=col, data=data, ax=ax)
#             ax.set_title(f"Gender vs {col}")
#             st.pyplot(fig)

# # 6. Hours Played vs Aggression
# st.subheader("6. Hours Played vs Aggression")

# # Standardize column names
# data.columns = data.columns.str.strip()

# time_mapping = {
#     "less than 1 hour": 0.5,
#     "more than 1 hour": 1.5,
#     "more than 2 hour": 2.5,
#     "more than 3 hour": 3.5,
#     "more than 5 hour": 5.5,
# }

# # Debugging: Check column names
# st.write("Column Names:", data.columns)

# # Correct column names based on your dataset
# hours_played_col = "How many hours do you play Video Games in  a day?"
# violent_hours_col = "How much time do you play \"violent\" video games specifically?"

# if all(col in data.columns for col in [hours_played_col, violent_hours_col]):
#     # Map time values to numeric
#     data["Hours_Played"] = data[hours_played_col].map(time_mapping)
#     data["Violent_Hours_Played"] = data[violent_hours_col].map(time_mapping)

#     # Drop rows with missing values
#     data = data.dropna(subset=["Hours_Played", "Violent_Hours_Played"])
#     if not data.empty:
#         fig, ax = plt.subplots()
#         sns.regplot(
#             x="Hours_Played",
#             y="Violent_Hours_Played",
#             data=data,
#             ax=ax,
#             line_kws={"color": "red"},
#         )
#         ax.set_title("Relationship Between Hours Played and Violent Hours Played")
#         ax.set_xlabel("Total Hours Played (numeric)")
#         ax.set_ylabel("Violent Hours Played (numeric)")
#         st.pyplot(fig)
#     else:
#         st.write("No valid data available after cleaning.")
# else:
#     st.write("Required columns for Hours Played analysis are missing.")


# # 7. Age vs Aggression Trends
# st.subheader("7. Age vs Aggression Trends")
# if "What is your age?" in data.columns and "Do you believe that playing violent video games can lead to aggressive behavior in real life?" in data.columns:
#     fig, ax = plt.subplots()
#     sns.lineplot(
#         x="What is your age?",
#         y="Do you believe that playing violent video games can lead to aggressive behavior in real life?",
#         data=data,
#         ax=ax,
#     )
#     ax.set_title("Age vs Belief in Aggression")
#     st.pyplot(fig)
# else:
#     st.write("Required columns for Age vs Aggression analysis are missing.")

# # 8. Behavioral Pairwise Analysis
# st.subheader("8. Behavioral Pairwise Analysis")
# response_mapping = {
#     "Strongly disagree": 1,
#     "Disagree": 2,
#     "Neither agree nor disagree": 3,
#     "Agree": 4,
#     "Strongly agree": 5,
# }
# behavior_cols = [
#     "Sometimes I lose temper for no good reason",
#     "I get into fights a little more than a normal person",
# ]
# valid_behavior_cols = [col for col in behavior_cols if col in data.columns]
# if len(valid_behavior_cols) < 2:
#     st.error("Not enough valid behavior columns for analysis. Ensure the data is complete.")
# else:
#     for col in valid_behavior_cols:
#         data[col] = data[col].map(response_mapping)
#     st.write(f"Analysis of: {valid_behavior_cols[0]}")
#     fig1, ax1 = plt.subplots()
#     sns.histplot(data[valid_behavior_cols[0]], bins=10, kde=True, ax=ax1)
#     ax1.set_title(valid_behavior_cols[0])
#     st.pyplot(fig1)
#     st.write(f"Analysis of: {valid_behavior_cols[1]}")
#     fig2, ax2 = plt.subplots()
#     sns.histplot(data[valid_behavior_cols[1]], bins=10, kde=True, ax=ax2)
#     ax2.set_title(valid_behavior_cols[1])
#     st.pyplot(fig2)
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.scatterplot(
#         x=valid_behavior_cols[0],
#         y=valid_behavior_cols[1],
#         data=data,
#         ax=ax,
#         alpha=0.7,
#     )
#     ax.set_title(f"Pairwise Analysis: {valid_behavior_cols[0]} vs {valid_behavior_cols[1]}")
#     st.pyplot(fig)




# # 9. Outlier Detection for Behavioral Features
# st.subheader("9. Outlier Detection for Behavioral Features")
# behavior_cols = [
#     "Sometimes I lose temper for no good reason",
#     "I get into fights a little more than a normal person",
# ]
# for col in behavior_cols:
#     fig, ax = plt.subplots()
#     sns.boxplot(data[col], ax=ax)
#     ax.set_title(f"Outlier Detection for {col}")
#     st.pyplot(fig)

# # 10. Game Type vs Aggression
# st.subheader("10. Game Type vs Aggression")
# # Define mapping for behavior categories
# def categorize_behavior(behavior):
#     if any(
#         keyword in behavior.lower()
#         for keyword in ["aggression", "aggressive", "anger", "angry", "fight", "furious", "violent"]
#     ):
#         return "Aggression"
#     elif any(
#         keyword in behavior.lower()
#         for keyword in ["relax", "fun", "happy", "excited", "good", "motivate"]
#     ):
#         return "Positive Effects"
#     elif any(
#         keyword in behavior.lower()
#         for keyword in ["stress", "frustration", "anxiety", "horrible", "tired", "bad"]
#     ):
#         return "Negative Effects"
#     elif any(
#         keyword in behavior.lower()
#         for keyword in ["no", "none", "nothing", "normal", "na", "nill"]
#     ):
#         return "No Change"
#     else:
#         return "Other"

# # Apply the categorization
# data["Behavior Category"] = data[
#     "What changes on behaviour have you experienced in yourself after playing violent video games?"
# ].fillna("No Response").apply(categorize_behavior)
# print(data["Behavior Category"].value_counts())
# if (
#     "What type of video games do you typically play?" in data.columns
#     and "Behavior Category" in data.columns
# ):
#     fig, ax = plt.subplots(figsize=(12, 6))
#     sns.countplot(
#         x="What type of video games do you typically play?",
#         hue="Behavior Category",
#         data=data,
#         ax=ax,
#     )
#     ax.set_title("Game Type vs Behavioral Categories")
#     ax.set_xlabel("Game Type")
#     ax.set_ylabel("Count")
#     plt.xticks(rotation=45)
#     st.pyplot(fig)
# else:
#     st.error("Required columns are missing or improperly processed.")



# # # 11. Family Type vs Behavior
# # response_mapping = {
# #     "Strongly disagree": 1,
# #     "Disagree": 2,
# #     "Neither agree nor disagree": 3,
# #     "Agree": 4,
# #     "Strongly agree": 5,
# # }
# # if "Type of Family" in data.columns and "Sometimes I lose temper for no good reason" in data.columns:
# #     data["Mapped Temper Response"] = data["Sometimes I lose temper for no good reason"].map(response_mapping)

# #     if data["Mapped Temper Response"].isnull().any():
# #         st.warning("Some responses in 'Sometimes I lose temper for no good reason' could not be mapped.")
# #     st.subheader("11. Family Type vs Behavior")
# #     fig, ax = plt.subplots()
# #     sns.boxplot(
# #         x="Type of Family",
# #         y="Mapped Temper Response",  # Use the mapped column for plotting
# #         data=data,
# #         ax=ax,
# #     )
# #     ax.set_title("Family Type vs Behavior")
# #     st.pyplot(fig)
# # else:
# #     st.error("Required columns are not present in the dataset.")


# # 12. Grouped Aggregations
# response_mapping = {
#     "Strongly disagree": 1,
#     "Disagree": 2,
#     "Neither agree nor disagree": 3,
#     "Agree": 4,
#     "Strongly agree": 5,
# }
# if "Gender" in data.columns:
#     data["Mapped Temper Response"] = data["Sometimes I lose temper for no good reason"].map(response_mapping)
#     data["Mapped Fight Response"] = data["I get into fights a little more than a normal person"].map(response_mapping)
#     grouped = data.groupby("Gender")[["Mapped Temper Response", "Mapped Fight Response"]].mean()
#     st.subheader("12. Grouped Aggregations")
#     st.write(grouped)
#     fig, ax = plt.subplots(figsize=(8, 5))
#     grouped.plot(kind='bar', ax=ax)
#     ax.set_title("Mean Responses by Gender")
#     ax.set_ylabel("Mean Response Score")
#     ax.set_xlabel("Gender")
#     ax.legend(["Temper Response", "Fight Response"])
#     st.pyplot(fig)


# # 13. Behavioral Clustering
# # Define behavior features
# behavior_features = [
#     "Sometimes I lose temper for no good reason", 
#     "I get into fights a little more than a normal person",
# ]
# st.subheader("13. Behavioral Clustering")

# # Check if all required columns are present
# if all(col in data.columns for col in behavior_features):
#     # Map categorical responses to numerical values
#     for feature in behavior_features:
#         data[f"Mapped {feature}"] = data[feature].map(response_mapping)
    
#     # Extract the mapped features for clustering
#     mapped_features = [f"Mapped {col}" for col in behavior_features]
#     X_cluster = data[mapped_features].dropna()
    
#     # Perform KMeans clustering
#     kmeans = KMeans(n_clusters=3, random_state=42)  # Fix: Import KMeans
#     clusters = kmeans.fit_predict(X_cluster)
#     data["Cluster"] = clusters  # Add cluster labels to the original data
    
#     # Display cluster centers and sample data
#     st.write("Cluster Centers:")
#     st.write(pd.DataFrame(kmeans.cluster_centers_, columns=behavior_features))
#     st.write("Clustered Data Sample:")
#     st.write(data[["Cluster"] + mapped_features].head())
    
#     # Cluster Visualization
#     st.subheader("Cluster Visualization")
#     sns.set_theme(style="whitegrid")
#     fig, ax = plt.subplots()
#     sns.scatterplot(
#         x=f"Mapped {behavior_features[0]}",
#         y=f"Mapped {behavior_features[1]}",
#         hue="Cluster",
#         palette="viridis",
#         data=data,
#         ax=ax
#     )
#     ax.set_title("Behavioral Clustering")
#     ax.set_xlabel(behavior_features[0])
#     ax.set_ylabel(behavior_features[1])
    
#     # Display the plot
#     st.pyplot(fig)
# else:
#     st.error("Required columns are not present in the dataset.")





# # 14. Behavioral Trait Comparison Across Age Groups
# st.subheader("14. Behavioral Trait Comparison Across Age Groups")

# if 'What is your age?' in data.columns:
#     # Convert 'What is your age?' to numeric and handle errors
#     data['What is your age?'] = pd.to_numeric(data['What is your age?'], errors='coerce')
    
#     data = data.dropna(subset=['What is your age?'])
    
#     # Create age groups
#     age_groups = pd.cut(
#         data['What is your age?'], 
#         bins=[0, 18, 25, 35, 50, np.inf], 
#         labels=['<18', '18-25', '26-35', '36-50', '50+']
#     )
#     data['Age Group'] = age_groups
#     behavioral_traits = ["I am a hot-tempered person", "Sometimes I lose temper for no good reason"]
#     for trait in behavioral_traits:
#         data[f"Mapped {trait}"] = data[trait].map(response_mapping)
#     for trait in behavioral_traits:
#         mapped_trait = f"Mapped {trait}"
#         if mapped_trait in data.columns:
#             st.subheader(f"{trait} Across Age Groups")
#             sns.set_theme(style="whitegrid")
#             fig, ax = plt.subplots()
#             sns.boxplot(x='Age Group', y=mapped_trait, data=data, ax=ax)
#             ax.set_title(f"{trait} Across Age Groups")
#             ax.set_ylabel(trait)
#             st.pyplot(fig)

response_mapping = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neither agree nor disagree": 3,
    "Agree": 4,
    "Strongly agree": 5,
}

# 16. Gaming Preferences by Gender
if "What type of video games do you typically play?" in data.columns and "Gender" in data.columns:
    sns.set_theme(style="whitegrid")
    st.write("Gaming Preferences by Gender")
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



# 17. Delinquent Behavior Trends
st.subheader("17. Delinquent Behavior Trends")

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


# 18. Aggression vs Residential Status
st.subheader("18. Aggression vs Residential Status")
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

# 19. Behavioral Trait Correlation Matrix
st.subheader("19. Behavioral Trait Correlation Matrix")
behavior_traits = [
    "I lose temper for no good reason",  # Correct column name
    "I get into fights a little more than a normal person",  # Correct column name
    "I am a hot-tempered person",  # Correct column name
    "Sometimes I lose temper for no good reason",  # Correct column name
]
if all(col in data.columns for col in behavior_traits):
    ig, ax = plt.subplots(figsize=(12, 8))
    trait_corr = data[behavior_traits].corr()
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

# 20. Aggression and Peer Perception
if "My friends say that I am a bit argumentative" in data.columns:
    st.subheader("20. Aggression and Peer Perception")
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

# 21. Video Game Influence on Delinquency
st.subheader("21. Video Game Influence on Delinquency")
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

# 22. Aggression by Family Type
st.subheader("22. Aggression by Family Type")
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
st.subheader("23. Gender and Aggression")
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
st.subheader("24. Aggression and Peer Conflict")
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
    st.subheader("25. Aggression Based on Game Type and Hours Played")
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