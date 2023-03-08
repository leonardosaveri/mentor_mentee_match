import pandas as pd
from collections import defaultdict
import streamlit as st

def match_mentors_and_mentees(mentors_df, mentees_df):
    # read in the data from the excel files
    
    mentors_df["name"] = mentors_df["Q1"] + ' ' + mentors_df["Q2"]

    mentees_df["name"] = mentees_df["Q1"] + ' ' +  mentees_df["Q2"]


    # create a defaultdict to store the mentors and their mentees
    mentor_dict = defaultdict(list)
    
    # loop through each mentee and find the mentor with the highest score
    for i, mentee_row in mentees_df.iterrows():
        best_mentors = []
        best_score = -1
        for j, mentor_row in mentors_df.iterrows():
            # check if this mentor already has 5 mentees
            if mentor_row['name'] in mentor_dict.keys() and len(mentor_dict[mentor_row['name']]) >= 5:
                continue
            
            # calculate the score for this mentor-mentee pair
            score = 0
            if mentee_row['Q4'] == mentor_row['Q4']:
                score += 10
            if mentee_row['Q5'] == mentor_row['Q5'] or mentee_row['Q6'] == mentor_row['Q6']:
                score += 5
            try:
                for interest in mentee_row['Q7'].split(','):
                    if interest in mentor_row['Q7']:
                        score += 1
            except:
                pass
            
            # update the best score and best mentors if applicable
            if score > best_score:
                best_score = score
                best_mentors = [mentor_row['name']]
            elif score == best_score:
                best_mentors.append(mentor_row['name'])
        
        # sort the best mentors by the number of mentees they already have
        best_mentors = sorted(best_mentors, key=lambda x: len(mentor_dict[x]))
        
        # add this mentee to the best mentor's list of mentees
        mentor_dict[best_mentors[0]].append(mentee_row['name'])
    
    # return the mentor-mentee dictionary
    return mentor_dict



# Define the app
def app():
    # Set the page title and description
    st.set_page_config(page_title='Mentor-Mentee List', page_icon=':clipboard:', layout='wide')
    st.title('Mentor-Mentee List')
    st.write('Upload two files (in .csv) to generate a list of mentors and their mentees.')

    # Create file uploader widgets
    file1 = st.file_uploader('Upload file Mentors', type=['csv', 'xlsx'])
    file2 = st.file_uploader('Upload file Mentees', type=['csv', 'xlsx'])

    # Check if files have been uploaded
    if file1 and file2:
        # Read the file data into a pandas dataframe
        mentor_df = pd.read_excel(file1).tail(-1) if file1.name.endswith('.xlsx') else pd.read_csv(file1, sep=';').tail(-1)
        st.write('Mentor:')
        st.write(mentor_df)

        mentees_df = pd.read_excel(file2).tail(-1) if file2.name.endswith('.xlsx') else pd.read_csv(file2, sep=';').tail(-1)
        st.write('Mentees:')
        st.write(mentees_df)

        # Call the function to get the mentor-mentee dictionary
        mentor_to_mentees = match_mentors_and_mentees(mentor_df, mentees_df)

        # Display the mentor-mentee dictionary
        st.header('Mentor-Mentee List')
        for mentor, mentees in mentor_to_mentees.items():
            st.write(f'**{mentor}**')
            mentor_data = mentor_df[mentor_df.name == mentor][['Q3', 'Q4', 'Q5', 'Q6', 'Q7']].values.tolist()[0]
            st.write(mentor_data[0])
            st.write(f'**Program:** {mentor_data[1]},   **Nationality:** {mentor_data[2]},  **High School:** {mentor_data[3]}')
            try:
                mentor_interests = ', '.join(mentor_data[4].split(','))
            except:
                mentor_interests = ''
            st.write(f'Interests: {mentor_interests}')
            st.write(mentees_df[mentees_df.name.isin(mentees)][['name', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']])
            st.write('---')

# Run the app
if __name__ == '__main__':
    app()
