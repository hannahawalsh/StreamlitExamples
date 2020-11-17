"""
This is an example of using Streamlit for machine learning.
It uses the Australian Rain data set from Kaggle to train random forests. 
"""

### Imports
import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
SEED = 101


def main():
    """
    This is the main body of the app and is what is run when the script is
    called from the terminal. It contains all the layout information and
    executes the appropriate computations.
    """
    ### Set up our side bar
    st.sidebar.title("Model Hyperparameters")
    help_button = st.sidebar.button(label="Help")
    n_trees = st.sidebar.number_input(label="Number of Trees in Forest",
                                      min_value=1, max_value=None, value=150)
    max_depth = st.sidebar.slider(label="Maximum Tree Depth", min_value=1,
                                  max_value=30, value=7)
    min_split = st.sidebar.selectbox(label="Minimum Samples Per Split",
                                     options=list(range(2, 11)), index=0)
    max_features = st.sidebar.radio(label="Feature Split Type",
                                    options=["Square Root", "Log (Base 2)",
                                    "All Features"], index=0)
    balanced = st.sidebar.checkbox(label="Balance Class Weight", value=False)
    start_button = st.sidebar.button(label="Train Model")


    ### Set up our main page 
    # Title
    html_title = ("<h1 style='text-align: center; font-size: {size}em; "
                  "font-family: Arial; color: DodgerBlue;'> Australian Rain "
                  "Predictions </h1>")
    st.markdown(html_title, unsafe_allow_html=True)
    
    # Add a free stock photo 
    pth = "https://image.freepik.com/free-photo/umbrella-rain_7186-1070.jpg"
    umbrella = st.image(pth, use_column_width=True)
    
    # Load in cached data and variables
    data = load_data()  
    cached = cached_values()
    model_dict = get_models()    
    
    # Add in placeholders
    close_help_spot = st.empty()
    help_text_spot = st.empty() 
    
    # Add a preview of the data 
    st.header("Rainfall Data")
    data_cont = st.beta_container()
    table_spot = data_cont.empty()
    b1, b2, _, _ = data_cont.beta_columns(4)
    minus_rows = b1.button("Fewer Rows")
    plus_rows = b2.button("More Rows")
    if not cached["display_data"]:
        cached["display_data"]["df"] = pd.concat([data["y_train"].head(5),
            data["X_train"].head(5)], axis=1).reset_index(drop=True)
    elif minus_rows:
        L = max(1, len(cached["display_data"]["df"]) - 1)
        cached["display_data"]["df"] = pd.concat([data["y_train"].head(L),
            data["X_train"].head(L)], axis=1).reset_index(drop=True)
    elif plus_rows:
        L = len(cached["display_data"]["df"]) + 1
        cached["display_data"]["df"] = pd.concat([data["y_train"].head(L),
            data["X_train"].head(L)], axis=1).reset_index(drop=True)
    sty_df = cached["display_data"]["df"].style.set_precision(2)
    table_spot.table(sty_df)

    # Add help information for hyperparameters 
    close_help = close_help_spot.button("Close")
    if help_button:
        cached["help_text_state"]["show"] = True
    if close_help:
        cached["help_text_state"]["show"] = False
    if cached["help_text_state"]["show"]:
        umbrella.empty()
        help_text_spot.markdown(cached["help_text"])
    else:
        close_help_spot.empty()
    
    # Train a random forest 
    if start_button:
        # Format inputs as propert sklearn parameters 
        mxf_dict = {"Square Root": "sqrt", "Log (Base 2)": "log2", 
                    "All Features": None}
        parameters = {"n_estimators": n_trees, "max_depth": max_depth,
                      "min_samples_split": min_split, 
                      "max_features": mxf_dict[max_features], 
                      "class_weight": "balanced" if balanced else None,
                      "random_state": SEED}
        model_name = "_".join([f"{k}={v}" for k, v in parameters.items()])
        
        if not model_name in model_dict:
            # Train and cache the model
            with st.spinner("Training Random Forest Model..."):
                model = RandomForestClassifier(**parameters)
                model.fit(data["X_train"], data["y_train"])
            model_dict["current_model"] = model 
            model_dict[model_name] = {"model": model, "parameters": parameters}
            
            results, cms = evaluate_model(model, data)
            cached["performance"]["metrics"] = results
            cached["performance"]["confusions"] = cms
            current_metrics = {"Train F1": results["Train"]["f1 score"],
                               "Test F1": results["Test"]["f1 score"],
                               "Train AUC": results["Train"]["AUC"],
                               "Test AUC": results["Test"]["AUC"]}
            colname = f"M{len(cached['past_metrics'].columns)+1}"
            cached["past_metrics"].insert(0, colname, 
                pd.Series({**current_metrics, **parameters}))
            
        else:
            # Use the previously trained model as the current model
            st.write("You trained this model before! Retrieving from cache.")
            model_dict["current_model"] = model_dict[model_name]["model"]
            
    
    if model_dict["current_model"]:
        ### Add information about the trained model
        st.markdown("## You have a trained model:")
        st.write(model_dict["current_model"])
        
        st.header("Model Performance:") 
        st.subheader("Metrics")
        st.write("How good is the model? You want these to be close to 1.")
        st.table(cached["performance"]["metrics"])
        
        st.subheader("Confusion Matrix")
        st.write("How did the model get it wrong?")
        col1, col2 = st.beta_columns(2)
        
        # col1.markdown("#### **Training Data**")
        col1.markdown("<h3 style='text-align: center'>Training Data</h3>",
                      unsafe_allow_html=True)
        col1.table(cached["performance"]["confusions"][0])
        col2.markdown("<h3 style='text-align: center'>Testing Data</h3>",
                      unsafe_allow_html=True)
        col2.table(cached["performance"]["confusions"][1])
        
    # Look at model performances for previously trained models 
    past = st.beta_expander("Trained Model Performances", False)
    def highlight_best(x):
        x2 = pd.DataFrame(index=x.index, columns=x.columns)
        max_col = x.loc["Test F1"].astype(float).idxmax()
        x2[max_col] = "background-color: yellow"
        return x2.fillna("background-color: white")
    if not cached["past_metrics"].empty:
        past.header("You've Trained These Models:")
        past_metrics = (cached["past_metrics"].style.set_precision(4)
            .apply(highlight_best, axis=None).format(None, na_rep="None"))
        past.dataframe(past_metrics)
    
    
@st.cache(suppress_st_warning=True, show_spinner=False)
def load_data():
    """ 
    Load the cleaned weather data from the csv. If the data hasn't been 
    downloaded, prompt a download. If it has been downloaded but not cleaned, 
    run the cleaning script. 
    """
    # Load the data from the sklearn package
    data_path = os.path.join("WeatherData", "cleaned_weather.csv")
    if not os.path.exists(data_path):
        original_path = os.path.join("WeatherData", "weatherAUS.csv")
        if not os.path.exists(original_path):
            st.error("Please download the Australian Weather data from "
                     "https://www.kaggle.com/jsphyg/weather-dataset-rattle-"
                     "package")
            st.stop()
        else:
            st.spinner("Cleaning Australian Weather Data")
            os.chdir("WeatherData")
            import clean_aus_weather
            os.chdir("..")
    data = pd.read_csv(data_path)
    y = data.pop("RainTomorrow")
    
    # Split it into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(data, y,
        test_size=0.15, random_state=SEED)
    data_dict =  {"X_train": X_train, "X_test": X_test, 
                  "y_train": y_train, "y_test": y_test}
    return data_dict

    
@st.cache(allow_output_mutation=True)
def cached_values():
    """ Save variables between runs """
    with open("./help_text.txt") as f:
        help_text = f.readlines()
    metric_idx = ["Train F1", "Test F1", "Train AUC", "Test AUC",
                  "n_estimators", "max_depth", "min_samples_split",
                  "max_features", "class_weight"]
    
    values = {"display_data": {},
              "help_text": "".join(help_text),
              "help_text_state": {"show": False},
              "performance": {"metrics": None, "confusions": None},
              "past_metrics": pd.DataFrame(index=metric_idx),
              }
    return values
    
    
@st.cache(allow_output_mutation=True)
def get_models():
    return {"current_model": None}



def evaluate_model(model, data):
    """
    Evaluate the random forest model on traning and testing data. 
    """
    y_hat_trn = model.predict(data["X_train"])
    y_hat_tst = model.predict(data["X_test"])
    
    # Custom classification report
    results = pd.DataFrame(index=["Train", "Test"])
    results["f1 score"] = [metrics.f1_score(data["y_train"], y_hat_trn),
                           metrics.f1_score(data["y_test"], y_hat_tst)] 
    results["accuracy"] = [metrics.accuracy_score(data["y_train"], y_hat_trn),
                           metrics.accuracy_score(data["y_test"], y_hat_tst)]
    fpr_trn, tpr_trn, _ = metrics.roc_curve(data["y_train"], y_hat_trn)
    fpr_tst, tpr_tst, _ = metrics.roc_curve(data["y_test"], y_hat_tst)
    results["AUC"] = [metrics.auc(fpr_trn, tpr_trn), 
                      metrics.auc(fpr_tst, tpr_tst)]
    
    # Confusion matrices    
    cols = ["Actually No Rain", "Actually Rain"]
    index = ["Predicts No Rain", "Predicts Rain"]
    confusion_trn = pd.DataFrame(metrics.confusion_matrix(data["y_train"],
                                 y_hat_trn), columns=cols, index=index)
    confusion_tst = pd.DataFrame(metrics.confusion_matrix(data["y_test"],
                                 y_hat_tst), columns=cols, index=index)
    return results.T, [confusion_trn, confusion_tst]



if __name__ == "__main__":
    main()