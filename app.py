import pandas as pd
from collections import defaultdict
import streamlit as st
from io import BytesIO



# streamlit_app.py

import streamlit as st

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
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
                try:
                    if 'BGL' in mentee_row['Q4'] and ('BIG' in mentor_row['Q4'] or 'CLMG' in mentor_row['Q4']):
                        score += 10
                    elif 'CLEACC' in mentee_row['Q4'] and 'BEMACC' in mentor_row['Q4']:
                        score += 10
                    elif mentee_row['Q4'] == mentor_row['Q4']:
                        score += 10
                except:
                    if mentee_row['Q4'] == mentor_row['Q4']:
                        score += 10
                if mentee_row['Q6'].lower().startswith('ita') and mentor_row['Q6'].lower().startswith('ita'):
                    score += 5
                elif mentee_row['Q6'] == mentor_row['Q6']:
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
        st.set_page_config(page_title='Matching Mentees to Mentors', page_icon=':clipboard:', layout='wide')
        st.title('Matching Mentees to Mentors')
        st.write('Upload here two files (in .xlsx or .csv) to generate a list of mentors and their mentees.')

        # Create file uploader widgets
        st.subheader('File Mentors')
        file1 = st.file_uploader('Upload file Mentors', type=['csv', 'xlsx'])
        if file1:
            mentor_df = pd.read_excel(file1).tail(-1) if file1.name.endswith('.xlsx') else pd.read_csv(file1, sep=';').tail(-1)
            if st.checkbox('Show Mentors'):
                st.write('Mentors:')
                st.write(mentor_df)

        
        st.subheader('File Mentees')
        file2 = st.file_uploader('Upload file Mentees', type=['csv', 'xlsx'])
        if file2:
            mentees_df = pd.read_excel(file2).tail(-1) if file2.name.endswith('.xlsx') else pd.read_csv(file2, sep=';').tail(-1)
            if st.checkbox('Show Mentees'):
                st.write('Mentees:')
                st.write(mentees_df)

        # Check if files have been uploaded
        if file1 and file2:
            # Call the function to get the mentor-mentee dictionary
            mentor_to_mentees = match_mentors_and_mentees(mentor_df, mentees_df)

            mentors_matches = pd.DataFrame(columns=['FIRSTNAME', 'EMAIL', 
                                                    'FRESHMAN_1', 'EMAIL_FRESHMEN_1', 
                                                    'FRESHMAN_2', 'EMAIL_FRESHMEN_2',
                                                    'FRESHMAN_3', 'EMAIL_FRESHMEN_3',
                                                    'FRESHMAN_4', 'EMAIL_FRESHMEN_4', 
                                                    'FRESHMAN_5', 'EMAIL_FRESHMEN_5'])

            mentees_matches = pd.DataFrame(columns=['FIRSTNAME', 'EMAIL', 
                                                    'NAME_MENTOR', 'EMAIL_MENTOR'])

            for mentor, mentees in mentor_to_mentees.items():
                mentor_data = mentor_df[mentor_df.name == mentor][['name', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q1']].values.tolist()[0]

                def mentee_output(mentees):
                    outputs = [mentor_data[-1], mentor_data[1]]
                    for mentee in mentees:
                        outputs.append(mentees_df[mentees_df.name == mentee][['Q1', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'name']].values.tolist()[0][-1])
                        outputs.append(mentees_df[mentees_df.name == mentee][['Q1', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'name']].values.tolist()[0][1])
                    if len(outputs) < 12:
                        outputs.extend([None] * (12 - len(outputs)))
                    return outputs
                mentors_matches.loc[len(mentors_matches)] = mentee_output(mentees)

                for mentee in mentees:
                    mentee_data = mentees_df[mentees_df.name == mentee][['Q1', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']].values.tolist()[0]
                    mentees_matches.loc[len(mentees_matches)] = mentee_data[0], mentee_data[1], mentor_data[0], mentor_data[1]

            
            st.header('Matches')
            if st.checkbox('Show Matches'):
                # Display the mentor-mentee dictionary
                st.header('Mentor-Mentee List')
                for mentor, mentees in mentor_to_mentees.items():
                    st.write(f'**{mentor}**')
                    mentor_data = mentor_df[mentor_df.name == mentor][['Q3', 'Q4', 'Q6', 'Q5', 'Q7']].values.tolist()[0]
                    st.write(mentor_data[0])
                    st.write(f'**Program:** {mentor_data[1]},   **Nationality:** {mentor_data[2]},  **High School:** {mentor_data[3]}')
                    try:
                        mentor_interests = ', '.join(mentor_data[4].split(','))
                    except:
                        mentor_interests = ''
                    st.write(f'Interests: {mentor_interests}')
                    st.table(mentees_df[mentees_df.name.isin(mentees)][['name', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']])
                    st.write('---')


            st.header('Download Matches')

            st.subheader('Mentees con Matches')
            # Create a download link to download the DataFrame as Excel
            excel_file = BytesIO()
            writer = pd.ExcelWriter(excel_file)
            mentees_matches.to_excel(writer, index=False)
            writer.save()
            excel_file.seek(0)
            st.download_button(label="Download mentees con matches.xlsx", data=excel_file, file_name='mentees con matches.xlsx', mime='application/vnd.ms-excel')

            st.subheader('Mentors con Matches')
        # Create a download link to download the DataFrame as Excel
            excel_file = BytesIO()
            writer = pd.ExcelWriter(excel_file)
            mentors_matches.to_excel(writer, index=False)
            writer.save()
            excel_file.seek(0)
            st.download_button(label="Download mentors con matches.xlsx", data=excel_file, file_name='mentors con matches.xlsx', mime='application/vnd.ms-excel')

    # Run the app
    if __name__ == '__main__':
        app()
