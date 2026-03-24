import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime
import glob
import numpy as np


st.set_page_config(
    page_title="Pilot Study on Masculine/Feminine Style Perception",
    layout="wide",
)

st.markdown(
    """
    <style>
    .header-large {
        font-size: 24px !important;
        font-weight: bold;
    }
    .custom-text {
        font-size: 17px !important;
        line-height: 1.6;
    }
    .custom-bold {
        font-size: 17px !important;
        font-weight: bold;
    }
    .custom-bullet {
        font-size: 17px !important;
        line-height: 1.6;
    }
    .pair-box {
        border: 1px solid #d9d9d9;
        border-radius: 10px;
        padding: 18px;
        min-height: 240px;
        background-color: #fafafa;
        font-size: 17px !important;
        line-height: 1.7;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    df = pd.read_csv("survey/ds_style_instructions.csv")
    idx = np.random.default_rng(42).choice(500, size=5, replace=False)
    df = df.iloc[idx]
    required_cols = {"feminine_style", "masculine_style"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in data_pairs.csv: {missing}")

    # Create an internal pair number
    df = df.reset_index(drop=True).copy()
    df["pair_id"] = df.index + 1

    # Optional columns
    if "is_attention_check" not in df.columns:
        df["is_attention_check"] = False
    if "label" not in df.columns:
        df["label"] = ""
    if "data" not in df.columns:
        df["data"] = ""

    return df


data = load_data()

if "responses" not in st.session_state:
    st.session_state["responses"] = [{} for _ in range(len(data))]

if "current_text_index" not in st.session_state:
    st.session_state["current_text_index"] = 0

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Page 1"

if "submitted" not in st.session_state:
    st.session_state["submitted"] = False

if "submitted_users" not in st.session_state:
    st.session_state["submitted_users"] = set()


def page1():
    st.title("Pilot Study on Masculine/Feminine Perception")

    st.markdown(
        '<p class="custom-text">We appreciate your feedback. Please fill out the survey below.</p>',
        unsafe_allow_html=True,
    )

    st.header("Consent Form")
    st.markdown(
        '<p class="custom-text">You are invited to participate in a pilot study designed to explore perceptions of contrasted linguistic style in written text. Before you decide to participate, it is important that you understand why this study is being conducted and what your participation involves. Please read the following information carefully.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="header-large">Description of the Research Study</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="custom-text">In this study, we aim to investigate what makes two texts sound different in gendered style, especially along a feminine–masculine dimension. By collecting human judgments about stylistic contrast, we hope to identify what factors influence one text of sounding more feminine or more masculine than another. For each pair of short texts, you will rate how strongly the two texts differ in feminine versus masculine style, choose which text sounds more feminine and which sounds more masculine, and judge how similar the two texts are in meaning/content and in fluency or grammar. Please focus on how the texts are written — such as their tone, word choice, and sentence structure — rather than what the texts are about. This research can help support future work on style transfer and AI-based writing assistance.</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="header-large">Consent</p>
        <p class="custom-text">Please indicate below that you are at least 18 years old, have read and understood this consent form, are comfortable using English to complete the task, and agree to participate in this research study.</p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        - I am 18 years old or older.
        - I have read this consent form or had it read to me.
        - My mother tongue is English.
        - I agree to participate in this research study and wish to proceed with the annotation task.
        """
    )

    current_consent = st.session_state.get("consent")
    consent_index = None
    if current_consent in ["I agree", "I do not agree"]:
        consent_index = ["I agree", "I do not agree"].index(current_consent)

    st.session_state["consent"] = st.selectbox(
        "If you give your consent to take part, please select 'I agree' below.",
        options=["I agree", "I do not agree"],
        index=consent_index,
        key="consent_selectbox",
        placeholder="Select an option",
    )

    if st.session_state.get("consent") == "I agree":
        if st.button("Next", key="page1_next"):
            st.session_state["current_page"] = "Page 2"
            st.rerun()
    elif st.session_state.get("consent") == "I do not agree":
        st.error("As you do not wish to participate in this study, please stop here.")
        st.button("Next", key="page1_next_disabled", disabled=True)
    else:
        st.button("Next", key="page1_next_empty", disabled=True)


def page2():
    st.session_state["p_id"] = st.text_input(
        "Please enter your Prolific ID",
        st.session_state.get("p_id", ""),
        key="prolific_id_input",
    )

    if st.button("Next", key="page2_next", disabled=not st.session_state.get("p_id")):
        if st.session_state["p_id"] == "hongyuchen":
            st.session_state["current_page"] = "Page 8"
        else:
            st.session_state["current_page"] = "Page 3"
        st.rerun()

    if st.button("Back", key="page2_back"):
        st.session_state["current_page"] = "Page 1"
        st.rerun()


def page3():
    st.header("Guidelines for Annotating Masculine/Feminine Style from Text Pairs")

    st.markdown(
        """
        <p class="custom-text">For each pair of texts, please judge how strongly the two texts differ in style and indicate which text sounds more feminine and which text sounds more masculine.</p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="custom-bold">Things to focus on:</p>
        <div class="custom-bullet">
            <ul>
                <li>overall tone</li>
                <li>word choice</li>
                <li>emotional vs direct style</li>
                
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
#<li>descriptive vs concise phrasing</li>
    if st.button("Next", key="page3_next"):
        st.session_state["current_page"] = "Page 4"
        st.rerun()

    if st.button("Back", key="page3_back"):
        st.session_state["current_page"] = "Page 2"
        st.rerun()


def page4():
    st.markdown('<p class="header-large">Examples</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Text A**")
        st.markdown(
            """
            <div class="pair-box">
            I couldn’t stop thinking about how kind and thoughtful her gesture was. It felt like a warm hug on a cold day.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("**Text B**")
        st.markdown(
            """
            <div class="pair-box">
            The machine operates at peak efficiency under optimal conditions. Ensure all components are calibrated before deployment.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <p class="custom-text"><strong>Example interpretation:</strong> The pair is strongly contrasted. Text A sounds more feminine. Text B sounds more masculine.</p>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Next", key="page4_next"):
        st.session_state["current_page"] = "Page 5"
        st.rerun()

    if st.button("Back", key="page4_back"):
        st.session_state["current_page"] = "Page 3"
        st.rerun()


def page5():
    st.header("Survey Instructions")

    st.markdown(
        f"""
        <p class="custom-text">
        There are {len(data[data["is_attention_check"] == False])} text pairs in this survey.
        For each pair, please judge how strongly the two texts differ in feminine versus masculine style,
        and indicate which text sounds more feminine and which sounds more masculine.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="custom-bullet">
            <ul>
                <li>There is no correct answer.</li>
                <li>Please follow your intuition.</li>
                <li>Base your judgment on style, not topic.</li>
                <li>Comments are optional.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Next", key="page5_next"):
        st.session_state["current_page"] = "Page 6"
        st.rerun()

    if st.button("Back", key="page5_back"):
        st.session_state["current_page"] = "Page 4"
        st.rerun()


def page6():
    st.header("Survey Questions")

    current_index = st.session_state["current_text_index"]

    try:
        row = data.iloc[current_index]
    except IndexError:
        st.error("Invalid index.")
        return

    text_a = row["feminine_style"]
    text_b = row["masculine_style"]
    is_attention_check = bool(row.get("is_attention_check", False))

    regular_pairs = data[data["is_attention_check"] == False]

    if is_attention_check:
        st.markdown("### Attention Check")
    else:
        regular_index = regular_pairs.index.get_loc(current_index) + 1
        st.markdown(f"### Pair {regular_index} of {len(regular_pairs)}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Text A")
        st.markdown(f'<div class="pair-box">{text_a}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### Text B")
        st.markdown(f'<div class="pair-box">{text_b}</div>', unsafe_allow_html=True)

    st.markdown("---")

    response = st.session_state["responses"][current_index]

    def update_contrast():
        st.session_state["responses"][current_index]["contrast"] = st.session_state.get(
            f"contrast_segmented_{current_index}"
        )

    def update_content_alignment():
        st.session_state["responses"][current_index]["content_alignment"] = st.session_state.get(
            f"content_segmented_{current_index}"
        )

    def update_grammar_alignment():
        st.session_state["responses"][current_index]["grammar_alignment"] = st.session_state.get(
            f"grammar_segmented_{current_index}"
        )

    contrast_options = [
        "1: Not contrasted at all",
        "2: Slightly contrasted",
        "3: Moderately contrasted",
        "4: Strongly contrasted",
        "5: Very strongly contrasted",
    ]

    content_options = [
        "1: Completely different",
        "2: Mostly different",
        "3: Partly similar",
        "4: Mostly same",
        "5: Same meaning",
    ]

    grammar_options = [
        "1: Very different",
        "2: Somewhat different",
        "3: Moderately similar",
        "4: Mostly same",
        "5: Same",
    ]

    feminine_options = ["Text A", "Text B", "About the same"]
    masculine_options = ["Text A", "Text B", "About the same"]

    confidence_options = [
        "1: Not Confident",
        "2: Somewhat Confident",
        "3: Moderately Confident",
        "4: Very Confident",
    ]

    contrast_value = response.get("contrast")
    content_value = response.get("content_alignment")
    grammar_value = response.get("grammar_alignment")

    st.markdown("**Style contrast**")
    contrast_kwargs = dict(
        label="How strongly contrasted are these two texts in feminine vs masculine style?",
        options=contrast_options,
        key=f"contrast_segmented_{current_index}",
        on_change=update_contrast,
    )
    if contrast_value in contrast_options:
        contrast_kwargs["default"] = contrast_value
    st.segmented_control(**contrast_kwargs)

    if response.get("contrast") is not None:
        st.write(f"Selected value: {response['contrast']}")
    else:
        st.write("No value selected yet.")

    st.markdown("**Content alignment**")
    content_kwargs = dict(
        label="To what extent do the two texts express the same meaning or content?",
        options=content_options,
        key=f"content_segmented_{current_index}",
        on_change=update_content_alignment,
    )
    if content_value in content_options:
        content_kwargs["default"] = content_value
    st.segmented_control(**content_kwargs)

    if response.get("content_alignment") is not None:
        st.write(f"Selected value: {response['content_alignment']}")
    else:
        st.write("No value selected yet.")

    st.markdown("**Grammar / fluency alignment**")
    grammar_kwargs = dict(
        label="To what extent do the two texts have the same level of fluency / grammatical acceptability?",
        options=grammar_options,
        key=f"grammar_segmented_{current_index}",
        on_change=update_grammar_alignment,
    )
    if grammar_value in grammar_options:
        grammar_kwargs["default"] = grammar_value
    st.segmented_control(**grammar_kwargs)

    if response.get("grammar_alignment") is not None:
        st.write(f"Selected value: {response['grammar_alignment']}")
    else:
        st.write("No value selected yet.")

    st.markdown("---")

    current_more_feminine = response.get("more_feminine")
    feminine_index = feminine_options.index(current_more_feminine) if current_more_feminine in feminine_options else None

    response["more_feminine"] = st.radio(
        "Which text sounds more feminine?",
        options=feminine_options,
        index=feminine_index,
        key=f"more_feminine_{current_index}",
    )

    current_more_masculine = response.get("more_masculine")
    masculine_index = masculine_options.index(current_more_masculine) if current_more_masculine in masculine_options else None

    response["more_masculine"] = st.radio(
        "Which text sounds more masculine?",
        options=masculine_options,
        index=masculine_index,
        key=f"more_masculine_{current_index}",
    )

    current_confidence = response.get("confidence")
    confidence_index = confidence_options.index(current_confidence) if current_confidence in confidence_options else None

    response["confidence"] = st.selectbox(
        "Confidence Level",
        options=confidence_options,
        index=confidence_index,
        key=f"confidence_{current_index}",
        placeholder="Select confidence",
    )

    response["comments"] = st.text_area(
        "Comments (Optional)",
        value=response.get("comments", ""),
        key=f"comments_{current_index}",
    )

    st.session_state["responses"][current_index] = response

    col_back, spacer, col_next = st.columns([1, 4, 1])

    with col_back:
        if st.button("Back", key=f"page6_back_{current_index}"):
            if current_index > 0:
                st.session_state["current_text_index"] -= 1
            else:
                st.session_state["current_page"] = "Page 5"
            st.rerun()

    with col_next:
        required_complete = (
            response.get("contrast") is not None
            and response.get("content_alignment") is not None
            and response.get("grammar_alignment") is not None
            and response.get("more_feminine") is not None
            and response.get("more_masculine") is not None
            and response.get("confidence") is not None
        )

        if st.button("Next", key=f"page6_next_{current_index}", disabled=not required_complete):
            if current_index < len(data) - 1:
                st.session_state["current_text_index"] += 1
            else:
                st.session_state["current_page"] = "Page 7"
            st.rerun()

    total_regular_pairs = len(data[data["is_attention_check"] == False])
    completed_regular_pairs = sum(
        1
        for i, r in enumerate(st.session_state["responses"])
        if not bool(data.iloc[i].get("is_attention_check", False))
        and r.get("contrast") is not None
        and r.get("content_alignment") is not None
        and r.get("grammar_alignment") is not None
        and r.get("more_feminine") is not None
        and r.get("more_masculine") is not None
        and r.get("confidence") is not None
    )

    progress = completed_regular_pairs / total_regular_pairs if total_regular_pairs else 0
    st.progress(progress)
    st.write(f"Completed {completed_regular_pairs} out of {total_regular_pairs} pairs.")
def page7():
    st.title("Your Feedback Matters")

    st.session_state["feedback"] = st.text_area(
        "Any questions, comments, or concerns?",
        value=st.session_state.get("feedback", ""),
        key="feedback_text_area",
    )

    if st.button("Next", key="page7_next"):
        st.session_state["current_page"] = "Page 8"
        st.rerun()

    if st.button("Back", key="page7_back"):
        st.session_state["current_page"] = "Page 6"
        st.rerun()


def page8():
    st.title("End of Survey")

    st.markdown(
        """
        Please click **Submit** to save your responses and receive your completion code.
        """,
    )

    if st.button("Submit", key="page8_submit", disabled=st.session_state.get("submitted", False)):
        user_id = f"{st.session_state.get('p_id', '')}"

        if user_id in st.session_state.get("submitted_users", set()):
            st.warning("You have already submitted the form.")
        else:
            responses_df = pd.DataFrame(st.session_state["responses"])
            responses_df["pair_id"] = data["pair_id"]
            responses_df["feminine_style"] = data["feminine_style"]
            responses_df["masculine_style"] = data["masculine_style"]
            responses_df["is_attention_check"] = data["is_attention_check"]
            responses_df["p_id"] = st.session_state.get("p_id", "")
            responses_df["feedback"] = st.session_state.get("feedback", "")
            responses_df["consent"] = st.session_state.get("consent", "")

            if "contrast" in responses_df.columns:
                responses_df["contrast_score"] = responses_df["contrast"].astype(str).str.split(":").str[0]
            else:
                responses_df["contrast_score"] = ""

            if "confidence" in responses_df.columns:
                responses_df["confidence_score"] = responses_df["confidence"].astype(str).str.split(":").str[0]
            else:
                responses_df["confidence_score"] = ""
            
            if "content_alignment" in responses_df.columns:
                responses_df["content_score"] = responses_df["content_alignment"].astype(str).str.split(":").str[0]
            else:
                responses_df["content_score"] = ""

            if "grammar_alignment" in responses_df.columns:
                responses_df["grammar_score"] = responses_df["grammar_alignment"].astype(str).str.split(":").str[0]
            else:
                responses_df["grammar_score"] = ""

            timestamp = int(time.time())
            submission_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"survey_responses_{user_id}_{submission_time}.csv"

            try:
                responses_df.to_csv(filename, index=False)
                st.success(
                    "Thank you for your submission!\n\n"
                   # "Submission code: **C1DSW210**"
                )
                st.session_state["submitted"] = True
                st.session_state["submitted_users"].add(user_id)
            except Exception as e:
                st.error(f"An error occurred while saving your response: {e}")

    if st.button("Back", key="page8_back"):
        st.session_state["current_page"] = "Page 7"
        st.rerun()

    user_id = f"{st.session_state.get('p_id', '')}"
    if user_id == "hongyuchen":
        st.markdown("---")
        st.header("Admin Section")

        password = st.text_input("Enter the password to download responses", type="password", key="admin_password")
        admin_password = os.getenv("arrsuccess", "arrsuccess")

        if password == admin_password:
            st.success("Password verified.")

            files = glob.glob("survey_responses_*.csv")
            if files:
                for file in files:
                    with open(file, "rb") as f:
                        st.download_button(
                            label=f"Download {file}",
                            data=f,
                            file_name=file,
                            mime="text/csv",
                            key=f"download_{file}",
                        )
            else:
                st.warning("No response files found.")
        elif password:
            st.error("Incorrect password.")


if st.session_state["current_page"] == "Page 1":
    page1()
elif st.session_state["current_page"] == "Page 2":
    page2()
elif st.session_state["current_page"] == "Page 3":
    page3()
elif st.session_state["current_page"] == "Page 4":
    page4()
elif st.session_state["current_page"] == "Page 5":
    page5()
elif st.session_state["current_page"] == "Page 6":
    page6()
elif st.session_state["current_page"] == "Page 7":
    page7()
elif st.session_state["current_page"] == "Page 8":
    page8()