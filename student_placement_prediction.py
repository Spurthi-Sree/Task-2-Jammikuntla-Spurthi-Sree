import streamlit as st # type: ignore[reportMissingImports]
import pandas as pd  # type: ignore[reportMissingImports]
import matplotlib.pyplot as plt # type: ignore[reportMissingImports]

from sklearn.model_selection import train_test_split  # type: ignore[reportMissingImports]
from sklearn.tree import DecisionTreeClassifier # type: ignore[reportMissingImports]
from sklearn.metrics import accuracy_score # type: ignore[reportMissingImports]

# -------------------------------------------
# Page Title
# -------------------------------------------

st.set_page_config(
    page_title="Student Placement Predication",
    page_icon="\U0001F393",
    layout="wide"
)

st.title("\U0001F393 Student Placement Prediction System")
st.write("Machine Learning Project Using Decision Tree Classifier")

# -------------------------------------------
# DataSet
# -------------------------------------------

data={
    "CGPA":
    [8.5,7.2,9.1,6.8,8.0,7.5,9.3,6.5,8.8,7.0],
    "Aptitude":
    [85,70,95,60,80,75,98,55,90,65],
    "Communication":
    [80,68,92,58,78,72,95,60,88,62],
    "Internship":
    [1,0,1,0,1,1,1,0,1,0],
    "Placed":
    [1,0,1,0,1,1,1,0,1,0]
}

df=pd.DataFrame(data)

# -------------------------------------------
# Show DataSet
# -------------------------------------------

st.subheader("\U0001F4CA Dataset")
st.dataframe(df)

# -------------------------------------------
# Visualization
# -------------------------------------------

st.subheader("\U0001F4C8 CGPA vs Aptitude")
fig,ax=plt.subplots()
ax.set_xlabel("CGPA")
ax.set_ylabel("Aptitude Score")
ax.set_title("CGPA vs Aptitude")

st.pyplot(fig)

# -------------------------------------------
# Machine Learning
# -------------------------------------------

X=df[["CGPA", "Aptitude", "Communication", "Internship"]]
y=df["Placed"]

X_train , X_test, y_train , y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
accuracy= accuracy_score(y_test,predictions)

st.subheader("\U0001F3AF Model Accuracy")
st.success(f"Accuracy: {accuracy*100:.2f}%")

# -------------------------------------------
# User Input
# -------------------------------------------

st.subheader("\U0001F4DD Enter Student Details")
cgpa = st.slider("CGPA", 0.0,10.0,7.5)
aptitude=st.slider("Aptitude Score",0,100,70)
communication=st.slider("Communication Score",0,100,70)

internship=st.selectbox(
    "Internship Experience",
    ["No","Yes"]
)

# -------------------------------------------
# Prediction
# -------------------------------------------

if st.button("Predict Placement"):
    internship_value= 1 if internship=="Yes" else 0
    result=model.predict([[cgpa,aptitude,communication,internship_value]])

    if result[0]==1:
        st.success("\u2705 Likely to be Placed")
    else:
        st.error("\u274C Likely NOT to be Placed")

# -------------------------------------------
# Bar Chart
# -------------------------------------------

st.subheader("\U0001F4CA Placement Distribution")
placement_count=df["Placed"].value_counts()

fig2,ax2=plt.subplots()

placement_count.plot(kind="bar",ax=ax2)

ax2.set_xlabel("0= Not Placed, 1= Placed")
ax2.set_ylabel("Number of Students")
ax2.set_title("Placement Statistics")

st.pyplot(fig2)

# -------------------------------------------
# Footer
# -------------------------------------------

st.markdown("---")
st.write("Developed using Python, Streamlit, Pandas, Scikit-learn and Matplotlib")