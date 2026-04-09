from ydata_profiling import ProfileReport


def generate_report(df):

    profile = ProfileReport(
        df,
        explorative=True
    )

    file_name = "EDA_Report.html"

    profile.to_file(file_name)

    return file_name