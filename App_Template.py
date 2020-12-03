import streamlit as st


### Define the main app here 
def main():
    
    # Set up your sidebar here 
    st.sidebar.title("The Sidebar")
    st.sidebar.header("Interactive Widgets:")
    
    button = st.sidebar.button("Button")
    checkbox = st.sidebar.checkbox("Checkbox", value=True)
    radio = st.sidebar.radio("Radio", options=[1, 2, 3], index=2)
    selectbox = st.sidebar.selectbox("Selectbox", options=["a", "b", "c"],
                                     index=1)
    mult_options = ["Hack", "to", "the", "Future"]
    multiselect = st.sidebar.multiselect("Multiselect", options=mult_options,
                                         default=mult_options)
    slider = st.sidebar.slider("Slider", min_value=-10, max_value=10, value=2,
                               step=1)
    select_options = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", 
                      "Friday", "Saturday"]
    select_slider = st.sidebar.select_slider("Select Slider",
        options=select_options, value=("Monday", "Friday"))
    
    
    
    # Set up your page here
    st.title("My Streamlit Application")
    "### _This_ is a 'magic' command using **markdown!**"
    content_spot = st.empty()
    st.header("More Interactive Widgets")
    
    
    text_input = st.text_input("Text Input")
    number_input = st.number_input("Number Input", value=0.75, min_value=0.0,
                                   max_value=1.0)
    text_area = st.text_area("Text Area", height=100)
    date_input = st.date_input("Date Input")
    time_input = st.time_input("Time Input")
    file_uploader = st.file_uploader("File Uploader")
    color_picker = st.color_picker("Color Picker", value="#C137A2")
    
    
    
    # Add content 
    past_values = cached_function()
    past_values.append(True)
    content_spot.markdown(f"<p style='color: {color_picker};'> You've taken "
                          f"{len(past_values)} actions </p>", 
                          unsafe_allow_html=True)


### Add our other functins here 
@st.cache(allow_output_mutation=True)
def cached_function():
        return []
    
    
def helping_function(arr):
    return " & ".join(arr)



    
if __name__ == "__main__":
    main()