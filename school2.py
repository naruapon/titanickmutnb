import pandas as pd
import matplotlib.pyplot as plt

# Set up Thai font for Matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Angsana New', 'Tahoma']

def count_majors_by_school(df):
    school_major_count = df.groupby('School')['Major'].count().reset_index()
    school_major_count.columns = ['School', 'Count']
    school_major_count = school_major_count.sort_values(by='Count', ascending=False).head(30)
    return school_major_count

def plot_school_major_count(school_major_count):
    plt.bar(school_major_count['School'], school_major_count['Count'])
    plt.xlabel('School')
    plt.ylabel('Major Count')
    plt.title('Top 30 Schools with the Most Majors')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load data from Excel file
    file_path = "science66-mod.xlsx"  # Replace with the actual file path
    df = pd.read_excel(file_path)

    school_major_count = count_majors_by_school(df)
    print(school_major_count)
    plot_school_major_count(school_major_count)
