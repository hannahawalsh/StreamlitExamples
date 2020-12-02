"""

"""
import altair as alt


def get_target_colors(clrs):
    col_dict = {0: "DarkMagenta", 1: "MediumOrchid", 2: "RebeccaPurple"}
    return [col_dict[c] for c in clrs]



def create_grouped_kde(df, col, target=None, by_class=True, size=175,
                       ylabel=None, xlabel=None):
    """ 
    Create an altair chart showing the kernel density estimate of a variable 
    with the option to plot a separate distribution by each class. 
    
    Parameters 
    ----------
    df : pandas DataFrame 
    col : string
        The column of df to be plotted 
    target : string
        The column of df that holds target labels 
    by_class : boolean, default True 
        Whether to plot data by class 
    size : integer 
        Size (width & height) of the returned plot 
    ylabel : string 
    xlabel : string 
    
    Returns 
    -------
    Altair chart 
    """
    ylabel = ylabel if ylabel is not None else "density"
    xlabel = xlabel if xlabel is not None else col
    
    labels = df[target].unique()
    
    if by_class:
        chart = alt.Chart(df, title=col).transform_density(
            density=col, counts=True, groupby=[target],
            steps=len(df), extent=[df[col].min() * 0.8,
            df[col].max() * 1.2], as_=[col, "density"]
        ).mark_area(
            opacity=0.7, 
            line=alt.OverlayMarkDef(stroke="black", strokeWidth=3)
        ).encode(
            x=alt.X(f"{col}:Q", title=xlabel),
            y=alt.Y("density:Q", title=ylabel),
            color=alt.Color(f"{target}:N", scale=alt.Scale(domain=labels,
                            range=get_target_colors(labels)), 
                            legend=alt.Legend(title="Wine classification"))
        )
    else:    
        chart = alt.Chart(df, title=col).transform_density(
            density=col, counts=True, steps=len(df),
            extent=[df[col].min() * 0.8, df[col].max() * 1.2],
            as_=[col, "density"]
        ).mark_area(
            opacity=0.7, 
            line=alt.OverlayMarkDef(stroke="black", strokeWidth=3)
        ).encode(
            x=alt.X(f"{col}:Q", title=xlabel),
            y=alt.Y("density:Q", title=ylabel),
            color=alt.value(get_target_colors(labels)[0])
        )
    return chart.properties(width=size, height=size)


def create_distribution_figure(df, by_class, size=175):
    """ 
    Create an altair chart for each column in a dataframe. Optionally, plot 
    distribution by a target class. 
    
    Parameters 
    ----------
    df : pandas DataFrame 
    by_class : boolean
        Whether to plot data by class 
    size : integer 
        Size (width & height) of the returned plot 
    
    Returns 
    -------
    Altair chart 
    """
    plot_rows = alt.vconcat(data=df)
    n_cols = 3
    n_rows = (len(df.columns) - 1) // n_cols + 1

    for i in range(0, len(df.columns), n_cols):
        current_row = df.columns[i: i+n_cols]
        plot_cols = alt.hconcat()
        target_labels = df.target.unique()
        
        for df_col in current_row:
            if df_col == "target":
                trg = df.target.value_counts().reset_index().rename(
                            columns={"target": "count", "index": "label"})
                cht = alt.Chart(trg).mark_bar(stroke="black", 
                    strokeWidth=3).encode(
                          x=alt.X("label:O", title="target"), 
                          y=alt.Y("count:Q", title="Count"), 
                          color=alt.Color("label:O", scale=alt.Scale(
                              domain=target_labels, 
                              range=get_target_colors(target_labels)), 
                              legend=None),
                          tooltip=["label", "count"]
                      ).properties(width=size, height=size, title="target")
                plot_cols |= cht 
            else:
                plot_cols |= create_grouped_kde(df, df_col, "target", by_class,
                                                size=size)
        plot_rows &= plot_cols
    chart = plot_rows.configure_legend(
            orient="top", titleFontSize=10,labelFontSize=10
        ).configure_title(fontSize=14, anchor="middle"
        ).configure_axis(grid=False, labelAngle=0
        )
    return chart


def create_comparison_figure(df, cols, by_class, size=100):
    """ 
    Create an altair chart that compairs up to 4 variables pairwise (immitates
    seaborn's pairplot).
    
    Parameters 
    ----------
    df : pandas DataFrame 
    cols : list
        The columns of df to be plotted 
    by_class : boolean, default True 
        Whether to plot data by class 
    size : integer 
        Size (width & height) of the returned plot 
    
    Returns 
    -------
    Altair chart 
    """
    plot_rows = alt.vconcat(data=df)
    labels = df["target"].unique()
    plot_colors = alt.Color(f"target:N", scale=alt.Scale(domain=labels,
                                range=get_target_colors(labels)))
    for i, col in enumerate(cols):
        plot_cols = alt.hconcat()        
        for j, secondary in enumerate(cols[:i+1]):
            ylabel = col if j == 0 else ""
            xlabel =  secondary if i == len(cols) - 1 else ""
            if secondary == col:
                cht = create_grouped_kde(df, col, "target", by_class=by_class,
                                         size=size, ylabel=ylabel,
                                         xlabel=xlabel)
                cht = cht.properties(title="")
            else:
                cht = alt.Chart(df).mark_circle(size=50
                    ).encode(x=alt.X(f"{secondary}:Q", title=xlabel), 
                             y=alt.Y(f"{col}:Q", title=ylabel),
                             color=plot_colors
                    ).properties(width=size, height=size)
            plot_cols |= cht
        plot_rows &= plot_cols
        
    chart = plot_rows.configure_axis(
            grid=False, titleFontSize=12
        ).configure_title(
        ).configure_legend(orient="top", titleFontSize=20, labelFontSize=16
        )
    if not by_class:
        return chart.configure_mark(color=get_target_colors(labels)[0])
    return chart


def create_corrolation_plot(data):
    """ 
    Create an altair chart that lists the correlation between variables and is 
    colored according to value. 
    
    Parameters 
    ----------
    data : pandas DataFrame 
    
    Returns 
    -------
    Altair chart 
    """
    corr = data.corr().reset_index().melt(id_vars="index")
    corr.columns = ["Variable 1", "Variable 2", "corr_values"]
    corr["Correlation"] = corr.corr_values.round(3)
    
    # Base chart 
    cht = alt.Chart(corr).encode(
            x="Variable 1:N", y="Variable 2:N"
        ).properties(title="Correlation Plot", width=700, height=700)
    
    # Text overlay 
    txt = cht.mark_text().encode(
            text="Correlation", 
            color=alt.condition(
                alt.datum.Correlation > 0.25, 
                if_true=alt.value('white'),
                if_false=alt.value('black')
        ))
    
    # Colored boxes 
    rct = cht.mark_rect().encode(
            color=alt.Color("corr_values:Q", scale=alt.Scale(scheme="purples"),
                            legend=None),
            tooltip=["Variable 1", "Variable 2", "Correlation"]
        )
    cht = rct + txt
    cht = cht.configure_axis(
            ticks=False, title=None, labelPadding=10, labelFontSize=14
        ).configure_title(
            fontSize=20, anchor="middle", dy=-20
        )
    return cht