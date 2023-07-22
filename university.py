import pandas as pd
import matplotlib.pyplot as plt

# Sample data: Replace this with your actual dataset or load it from a file
data = {
    'Major': ['Computer Science', 'Electrical Engineering', 'Biology', 'Psychology', 'Physics'],
    'Faculty': ['Engineering', 'Engineering', 'Science', 'Arts', 'Science']
}

# Convert data to a DataFrame
df = pd.DataFrame(data)

def count_majors_by_faculty(df):
    faculty_major_count = df.groupby('Faculty')['Major'].count().reset_index()
    faculty_major_count.columns = ['Faculty', 'Count']
    return faculty_major_count

def plot_faculty_major_count(faculty_major_count):
    plt.bar(faculty_major_count['Faculty'], faculty_major_count['Count'])
    plt.xlabel('Faculty')
    plt.ylabel('Major Count')
    plt.title('Number of Majors in Each Faculty')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    faculty_major_count = count_majors_by_faculty(df)
    print(faculty_major_count)
    plot_faculty_major_count(faculty_major_count)
