import pandas as pd
import matplotlib.pyplot as plt

# Set up Thai font for Matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Angsana New', 'Tahoma']

def find_top_school(df):
    school_major_count = df.groupby('School')['Major'].count().reset_index()
    school_major_count.columns = ['School', 'Major Count']
    top_school = school_major_count.sort_values(by='Major Count', ascending=False).iloc[0]['School']
    return top_school

def count_majors_by_school(df, school_name):
    school_major_count = df[df['School'] == school_name]['Major'].value_counts().reset_index()
    school_major_count.columns = ['Major', 'Count']
    return school_major_count

def plot_school_major_count(school_major_count, school_name):
    plt.barh(school_major_count['Major'], school_major_count['Count'])
    plt.xlabel('จำนวน')
    plt.ylabel('สาขา')
    plt.title(f'จำนวนสาขาใน{school_name}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load data from Excel file
    file_path = "science66-mod.xlsx"  # Replace with the actual file path
    df = pd.read_excel(file_path)

    top_school_name = find_top_school(df)

    school_major_count = count_majors_by_school(df, top_school_name)
    print(school_major_count)
    plot_school_major_count(school_major_count, top_school_name)
