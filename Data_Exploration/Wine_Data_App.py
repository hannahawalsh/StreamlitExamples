"""
This is an example of using Streamlit for data exploration.
It uses the built-in sklearn wine data set.
"""

### Imports
import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine

from AppPlotFunctions import *



### Main application function
def main():
    """
    This is the main body of the app and is what is run when the script is
    called from the terminal. It contains all the layout information and
    executes the appropriate computations.
    """
    ### Load the data
    data = load_data()
    data_cols = list(data.columns)

    ### Set up our app layout
    # Main page title: HTML formatting
    html_page_title = ("<h1 style='text-align: center; font-size: 3.0em; "
                       "color: RebeccaPurple;'> Wine Data Exploration </h1>")
    st.markdown(html_page_title, unsafe_allow_html=True)

    # Main page: use containers
    data_container = st.beta_container()
    filter_container = st.beta_expander("Data Filtering", expanded=False)
    corrolation_map = st.beta_expander("Correlation Plot", expanded=True)
    distribution_plot = st.beta_expander("Distribution Plot", expanded=True)
    comparison_plot = st.beta_expander("Comparison Plot", expanded=True)
    description = st.beta_expander("Data Description", expanded=False)
    description.write(return_description())

    # Sidebar: use containers
    display_options = st.sidebar.beta_container()
    st.sidebar.write("---")
    plot_options = st.sidebar.beta_container()


    ### Add content to the app
    ## To sidebar
    display_options.title("Display Options")
    colr_data = display_options.checkbox("Highlight data based on class")
    sort_cont = display_options.beta_columns([4, 3])
    sort_col = sort_cont[0].selectbox("Sort:", ["---"] + data_cols,
                                      key="sort_by")
    sort_cont[0].empty()
    ascending = sort_cont[1].selectbox(" ­­", ["Low->High", "High->Low"],
                                       key="sort_how")
    num_rows = display_options.number_input("Number of Rows to Display", 1,
                                            data.shape[0], 5)
    plot_options.title("Plotting Options")
    by_class = plot_options.checkbox("Show distribution by class", True)
    plot_filtered = plot_options.checkbox("Plot filtered data", True)
    plot_options.write("Select 4 Features to Compare:")
    plot_vars = get_plot_vars()[-4:]
    plot_vars.append(plot_options.selectbox("Feature 1", ["---"] + data_cols,
                                  index=1, key="f1"))
    plot_vars.append(plot_options.selectbox("Feature 2", ["---"] +
                     exclude(data_cols, plot_vars), index=1, key="f2"))
    plot_vars.append(plot_options.selectbox("Feature 3", ["---"] +
                     exclude(data_cols, plot_vars), index=1, key="f3"))
    plot_vars.append(plot_options.selectbox("Feature 4", ["---"] +
                     exclude(data_cols, plot_vars), index=1, key="f4"))

    ## To the main page
    #  Filters
    filter_container.markdown("##### Add a new filter:")
    filter_cols = filter_container.beta_columns([2, 1, 1])
    col_filter = filter_cols[0].selectbox("", data_cols)
    filter_exp = filter_cols[1].selectbox("", ["<", "==", ">"], index=2)
    filter_num = filter_cols[2].number_input("", data[col_filter].min(),
                    data[col_filter].max(), data[col_filter].mode()[0])
    add_filter = filter_container.button("Add filter")

    current_filters = filter_container.beta_columns(2)
    current_filters[0].markdown("#### Staged filters:")
    staged_text = current_filters[0].empty()
    apply_filters = current_filters[0].button("Apply Filters")
    current_filters[1].markdown("#### Active filters:")
    active_text = current_filters[1].empty()
    rm_filters = current_filters[1].button("Remove all filters")

    # Retrieve all set filters and the previously filtered frame
    active_filters = get_filters()
    filtered_data = get_filtered_data()
    filtered =  (filtered_data["data"] if filtered_data["data"] is not None
                 else data.copy())

    # Add new filter button
    if add_filter:
        if col_filter in active_filters:
            subdict = active_filters[col_filter]
            active_filters.update({col_filter: get_filter_update(subdict,
                                  filter_exp, filter_num)})
        else:
            subdict = {"<": None, ">": None, "==": None, "active": False}
            subdict[filter_exp] = filter_num
            active_filters.update({col_filter: subdict})

    # Apply filters button
    if apply_filters:
        queries = []
        for col, filter_dict in active_filters.items():
            queries.append(get_filter_expression(filter_dict, col))
            active_filters[col].update({"active": True})
        if queries:
            filtered = filtered.query(" & ".join(queries))

    # Remove filters button
    if rm_filters:
        active_filters.clear()  # clears the cache
        filtered = data.copy()

    # Display what filters have been "staged" and which ones are "active"
    staged_text.markdown("\---")
    active_text.markdown("\---")
    if active_filters:
        staged = [get_filter_expression(filter_dict, col) for col, filter_dict
                  in active_filters.items() if "active" in filter_dict.keys()
                  and filter_dict["active"] == False]
        active = [get_filter_expression(filter_dict, col) for col, filter_dict
                  in active_filters.items() if "active" in filter_dict.keys()
                  and filter_dict["active"] == True]
        if staged:
            staged_text.markdown("  \n".join(staged))
        if active:
            active_text.markdown("  \n".join(active))

    # Cache the newly filtered frame
    filtered_data.update({"data": filtered})

    # Display the filtered & styled data
    Style.num_rows = num_rows
    Style.colored = True if colr_data else False
    Style.sort_column = sort_col
    Style.ascending = True if ascending == "Low to High" else False
    data_container.write(f"Dataframe contains {filtered.shape[0]} data points")
    # data_container.table(Style().style(filtered))
    data_container.table(Style().style(filtered))

    # Plots
    plot_df = filtered if plot_filtered else data
    corrolation_map.altair_chart(create_corrolation_plot(plot_df),
                                 use_container_width=True)
    distribution_plot.altair_chart(create_distribution_figure(plot_df,
                                                              by_class))
    compare_cols = [v for v in plot_vars if v != "---"]
    if compare_cols:
        comparison_plot.altair_chart(create_comparison_figure(plot_df,
                                     compare_cols, by_class, size=120))



### Cached functions
@st.cache(allow_output_mutation=True)
def load_data():
    """ Load and shuffle the sklearn wine data set. """
    data = load_wine(as_frame=True).frame
    return data.sample(frac=1, replace=False,
                       random_state=101).reset_index(drop=True)

@st.cache(allow_output_mutation=True)
def return_description():
    return load_wine().DESCR[19:]

@st.cache(allow_output_mutation=True)
def get_plot_vars():
    return []

@st.cache(allow_output_mutation=True)
def get_filters():
    return {}

@st.cache(allow_output_mutation=True)
def get_filtered_data():
    return {"data": None}


### Other Functions
def exclude(array, exclude_things):
    """ For a given array, return that array without the 'exclude_things'"""
    if not isinstance(exclude_things, list):
        exclude_things = [exclude_things]
    return [x for x in array if x not in exclude_things]

def get_filter_update(subdict, filter_exp, filter_num):
    """ Gets the update to the filter dictionary """
    if filter_exp == "==":
        subdict = {"<": None, ">": None, "==": filter_num, "active": False}
    else:
        subdict.update({filter_exp: filter_num, "==": None, "active": False})
        other = "<" if filter_exp == ">" else ">"
        if subdict[">"] and subdict["<"] and (subdict[">"] >  subdict["<"]):
            subdict[other] = None
    return subdict

def get_filter_expression(filter_dict, col):
    """ Get the expresion for the filter """
    defined = [k for k, v in filter_dict.items() if v and k!="active"]
    return " and ".join([f"{col} {k} {round(filter_dict[k], 2)}" for
                         k in defined])


### Style class to display dataframe
class Style:
    num_rows = 3
    colored = False
    sort_column = "---"
    ascending = True

    def color_class(self, row):
        """ Color each row according to it's target variable """
        colors = ["DarkMagenta", "MediumOrchid", "RebeccaPurple"]
        return [f"background-color: {colors[int(row.target)]}"]*len(row)


    def style(self, df):
        """ Apply all the styles to a dataframe df """
        ### Change Style
        # Display 2 decimal places
        if Style.sort_column != "---":
            df = df.sort_values(by=Style.sort_column,
                                ascending=Style.ascending)
        st_df = df.head(Style.num_rows).style.set_precision(2)
        font_color = "black"
        if Style.colored == True:
            st_df = st_df.apply(self.color_class, axis="columns")
            font_color = "white"

        return st_df.set_properties(**{"color": font_color})



if __name__ == "__main__":
    ### Set the color palette
    colors = ["DarkMagenta", "MediumOrchid", "RebeccaPurple", "DarkOrchid",
              "DarkViolet", "BlueViolet", "Indigo", "MediumPurple", "Purple"]
    main()
