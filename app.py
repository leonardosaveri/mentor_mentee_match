# streamlit_app.py

import re
import pandas as pd
import numpy as np
from collections import defaultdict
from io import BytesIO
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# ────────── 1. CENTRALIZE COLUMN NAMES HERE ──────────
COL_FULLNAME    = "Full name"
COL_EMAIL       = "Email Address"
COL_COURSE      = "Bachelor's Degree"
COL_HS_COUNTRY  = "High School Country"
COL_CITIZENSHIP = "Citizenship"
COL_LANGUAGES   = "Which languages do you speak fluently?"
COL_INTERESTS   = "Hobbies and passions (please, list up to 4)"

REQUIRED_COLUMNS = [
    COL_FULLNAME, COL_EMAIL, COL_COURSE,
    COL_HS_COUNTRY, COL_CITIZENSHIP,
    COL_LANGUAGES, COL_INTERESTS
]

# ────────── 2. PASSWORD CHECK ──────────
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets.get("password"):  # secure check
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    if not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    return True

# ────────── 3. MATCHING FUNCTION WITH WEIGHTED SCORES ──────────
def match_mentors_and_mentees(mentors_df: pd.DataFrame, mentees_df: pd.DataFrame) -> dict:
    from collections import defaultdict

    mentors_df = (
        mentors_df[mentors_df["Email Address"].str.contains("@", na=False)]
        .sort_values(COL_FULLNAME)
        .drop_duplicates(COL_EMAIL, keep="first")
        .reset_index(drop=True)
    )

    mentees_df = (
        mentees_df[mentees_df["Email Address"].str.contains("@", na=False)]
        .sort_values(COL_FULLNAME)  # optional: choose which name to keep (e.g., alphabetically)
        .drop_duplicates(COL_EMAIL, keep="first")
        .reset_index(drop=True)
    )


    mentors_df[COL_COURSE] = mentors_df[COL_COURSE].replace("CLEF", "BIEF")

    mentors = mentors_df.copy().reset_index(drop=True)
    mentees = mentees_df.copy().reset_index(drop=True)

    mentors["Course_l"] = mentors[COL_COURSE].fillna("").str.lower()
    mentees["Course_l"] = mentees[COL_COURSE].fillna("").str.lower()
    mentors["HS_l"] = mentors[COL_HS_COUNTRY].fillna("").str.lower()
    mentees["HS_l"] = mentees[COL_HS_COUNTRY].fillna("").str.lower()
    mentors["Res_l"] = mentors[COL_CITIZENSHIP].fillna("").str.lower()
    mentees["Res_l"] = mentees[COL_CITIZENSHIP].fillna("").str.lower()

    def parse_languages(s):
        if not isinstance(s, str):
            return set()
        return {
            lang for lang in re.split(r"[;,]\s*", s.lower().strip())
            if lang and lang != "Other"
        }


    mentors["Lang_set"] = mentors[COL_LANGUAGES].apply(parse_languages)
    mentees["Lang_set"] = mentees[COL_LANGUAGES].apply(parse_languages)

    # TF-IDF similarity
    all_interests = pd.concat([
        mentors[COL_INTERESTS].fillna(""),
        mentees[COL_INTERESTS].fillna("")
    ])
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(all_interests.tolist())
    sim_matrix = cosine_similarity(tfidf_matrix[:len(mentors)], tfidf_matrix[len(mentors):])

    assignments = defaultdict(list)
    assigned_mentees = set()
    mentor_capacity = {mentor["Full name"]: 0 for _, mentor in mentors.iterrows()}

    # STEP 1: Ensure one mentee per mentor (initial assignment)
    mentees_remaining = mentees.copy()
    for mentor_idx, mentor in mentors.iterrows():
        mentor_name = mentor["Full name"]
        best_score = -1
        best_idx = None
        for mentee_idx, mentee in mentees_remaining.iterrows():
            if not(mentor["Lang_set"] & mentee["Lang_set"]):
                continue
            score = 0
            if mentor["Course_l"] == mentee["Course_l"]:
                score += 10
            if mentor["HS_l"] == mentee["HS_l"]:
                score += 3
            if mentor["Res_l"] == mentee["Res_l"]:
                score += 2
            score += 4 * sim_matrix[mentor_idx, mentee_idx]

            if score > best_score:
                best_score = score
                best_idx = mentee_idx

        if best_idx is not None:
            best_mentee = mentees_remaining.loc[best_idx]
            assignments[mentor_name].append(best_mentee["Full name"])
            assigned_mentees.add(best_mentee["Full name"])
            mentor_capacity[mentor_name] += 1
            mentees_remaining = mentees_remaining.drop(best_idx)

    # STEP 2: Assign remaining mentees — try to balance (max 3 or 4 per mentor)
    max_per_mentor = len(mentees) // len(mentors) + 1
    for mentee_idx, mentee in mentees.iterrows():
        if mentee["Full name"] in assigned_mentees:
            continue

        best_score = -1
        best_mentor = None
        for mentor_idx, mentor in mentors.iterrows():
            mentor_name = mentor["Full name"]
            if mentor_capacity[mentor_name] >= max_per_mentor:
                continue

            score = 0
            if mentor["Course_l"] == mentee["Course_l"]:
                score += 10
            if mentor["HS_l"] == mentee["HS_l"]:
                score += 3
            if mentor["Res_l"] == mentee["Res_l"]:
                score += 2
            score += 5 * sim_matrix[mentor_idx, mentee_idx]

            if score > best_score:
                best_score = score
                best_mentor = mentor_name

        if best_mentor:
            assignments[best_mentor].append(mentee["Full name"])
            mentor_capacity[best_mentor] += 1
            assigned_mentees.add(mentee["Full name"])

    # STEP 3: Final fallback — assign remaining mentees to any mentor with space
    for mentee_idx, mentee in mentees.iterrows():
        if mentee["Full name"] in assigned_mentees:
            continue
        for mentor_name in mentor_capacity:
            if mentor_capacity[mentor_name] < max_per_mentor:
                assignments[mentor_name].append(mentee["Full name"])
                mentor_capacity[mentor_name] += 1
                assigned_mentees.add(mentee["Full name"])
                break

    return assignments






# ────────── 4. STREAMLIT UI ──────────
def app():
    st.set_page_config(page_title="Matching Mentees to Mentors", layout="wide")
    st.title("📋 Matching Mentees to Mentors")
    st.markdown("**Upload** your Mentor/Mentee files. Preview shows only core columns.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mentors")
        f1 = st.file_uploader("Choose Mentors (xlsx/csv)", type=["xlsx","csv"], key="m1")
        if f1:
            mdf = pd.read_excel(f1) if f1.name.lower().endswith("xlsx") else pd.read_csv(f1)
            cols = [c for c in REQUIRED_COLUMNS if c in mdf.columns]
            with st.expander("Preview Mentors"):
                st.dataframe(mdf[cols])
    with col2:
        st.subheader("Mentees")
        f2 = st.file_uploader("Choose Mentees (xlsx/csv)", type=["xlsx","csv"], key="m2")
        if f2:
            ddf = pd.read_excel(f2) if f2.name.lower().endswith("xlsx") else pd.read_csv(f2)
            cols = [c for c in REQUIRED_COLUMNS if c in ddf.columns]
            with st.expander("Preview Mentees"):
                st.dataframe(ddf[cols])

    if "mdf" in locals() and "ddf" in locals():
        miss_m = [c for c in REQUIRED_COLUMNS if c not in mdf.columns]
        miss_d = [c for c in REQUIRED_COLUMNS if c not in ddf.columns]
        if miss_m or miss_d:
            st.error(f"Missing columns:\n Mentors: {miss_m}\n Mentees: {miss_d}")
            return
        with st.spinner("Computing matches…"):
            result = match_mentors_and_mentees(mdf, ddf)

        mentors_cols = [COL_FULLNAME, COL_EMAIL,
                        "Mentee 1","Email Mentee 1",
                        "Mentee 2","Email Mentee 2",
                        "Mentee 3","Email Mentee 3",
                        "Mentee 4","Email Mentee 4",
                        "Mentee 5","Email Mentee 5"]
        mentees_cols = [COL_FULLNAME, COL_EMAIL, "Assigned Mentor","Mentor Email"]

        mentors_out = pd.DataFrame(columns=mentors_cols)
        mentees_out = pd.DataFrame(columns=mentees_cols)
        for mentor, mentees_list in result.items():
            mrow = mdf[mdf[COL_FULLNAME]==mentor].iloc[0]
            ro = {COL_FULLNAME: mentor, COL_EMAIL: mrow[COL_EMAIL]}
            for i, name in enumerate(mentees_list[:5],1):
                prow = ddf[ddf[COL_FULLNAME]==name].iloc[0]
                ro[f"Mentee {i}"]=name; ro[f"Email Mentee {i}"]=prow[COL_EMAIL]
            mentors_out.loc[len(mentors_out)] = [ro.get(c) for c in mentors_cols]
            for name in mentees_list:
                prow=ddf[ddf[COL_FULLNAME]==name].iloc[0]
                mentees_out.loc[len(mentees_out)]=[name,prow[COL_EMAIL],mentor,mrow[COL_EMAIL]]

        st.header("🏷️ Matches Summary")
        with st.expander("Mentor → Mentees"):
            for ment, menl in result.items():
                # Mentor info
                mrow = mdf[mdf[COL_FULLNAME]==ment].iloc[0]
                # Display mentor summary inline
                info = (
                    f"Email: *{mrow[COL_EMAIL]}*, "
                    f"Course: *{mrow[COL_COURSE]}*, "
                    f"HS: *{mrow[COL_HS_COUNTRY]}*, "
                    f"Residence: *{mrow[COL_CITIZENSHIP]}*, "
                    f"Languages: *{mrow[COL_LANGUAGES]}*, "
                    f"Interests: *{mrow[COL_INTERESTS]}*"
                )
                st.write(f"**{ment}** ({len(menl)} mentees) — ({info})")
                st.table(ddf[ddf[COL_FULLNAME].isin(menl)][REQUIRED_COLUMNS])
                st.markdown("---")

        with st.expander("Mentees → Mentor"):
            st.dataframe(mentees_out)

        # Downloads
        buf1=BytesIO();
        with pd.ExcelWriter(buf1,engine="xlsxwriter") as w: mentors_out.to_excel(w,index=False)
        buf1.seek(0)
        st.download_button("Download Mentors_with_Mentees.xlsx",buf1,"Mentors_with_Mentees.xlsx")
        buf2=BytesIO();
        with pd.ExcelWriter(buf2,engine="xlsxwriter") as w: mentees_out.to_excel(w,index=False)
        buf2.seek(0)
        st.download_button("Download Mentees_with_Mentor.xlsx",buf2,"Mentees_with_Mentor.xlsx")

if __name__ == "__main__":
    if check_password():
        app()
