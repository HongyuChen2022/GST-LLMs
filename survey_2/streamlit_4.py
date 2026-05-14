import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime
import glob


# ============================================================
# CONFIGURATION
# ============================================================

N_SURVEY_VERSIONS = 5
SURVEY_VERSION_FOLDER = "survey_versions"
COMPLETION_CODE = "C1DSW210"  # replace with your real Prolific completion code
ADMIN_ID = "hongyuchen"

STYLE_OPTIONS = [
    "1: Very Feminine",
    "2: Somewhat Feminine",
    "3: Neutral",
    "4: Somewhat Masculine",
    "5: Very Masculine",
]

CONFIDENCE_OPTIONS = [
    "1: Not Confident. You were unsure or found the text ambiguous",
    "2: Somewhat Confident. You made a judgment but still felt uncertain or had significant doubts",
    "3: Moderately Confident. You felt reasonably sure of your judgment but had some doubts",
    "4: Very Confident. You were very certain about your judgment with no hesitation",
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .header-large {
        font-size: 24px !important;
        font-weight: bold;
    }
    .custom-text {
        font-size: 17px !important;
    }
    .custom-bold {
        font-size: 17px !important;
        font-weight: bold;
    }
    .custom-bullet {
        font-size: 17px !important;
    }
    .stSelectbox {
        font-size: 17px !important;
    }
    .custom-label {
        font-size: 17px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data(version):
    """
    Loads one pre-generated survey version.

    Expected file path:
    survey_versions/version_0.csv
    survey_versions/version_1.csv
    ...
    survey_versions/version_4.csv

    Expected columns:
    - short_text
    - is_attention_check
    - expected_answer
    - item_id
    - data_source
    - source_dataset
    - style_condition
    """
    path = f"survey_2/{SURVEY_VERSION_FOLDER}/version_{version}.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find survey version file: {path}")

    data = pd.read_csv(path)

    if "is_attention_check" not in data.columns:
        data["is_attention_check"] = False

    if "expected_answer" not in data.columns:
        data["expected_answer"] = ""

    return data


def get_version_from_url():

    try:
        version = int(st.query_params.get("version", 4))
    except Exception:
        version = 4

    if version < 0 or version >= N_SURVEY_VERSIONS:
        st.error(
            f"Invalid survey version: {version}. "
            f"Please use a version between 0 and {N_SURVEY_VERSIONS - 1}."
        )
        st.stop()

    return version


def is_attention_check_value(value):
    """
    Handles True/False values that may be saved as booleans or strings in CSV.
    """
    return str(value).lower() == "true"


# ============================================================
# PAGE 1: CONSENT
# ============================================================

def page1():
    st.title("Pilot Study on Masculine/Feminine/Gender-Neutral Style Perception")

    st.header("Consent Form")

    st.markdown(
        '<p class="custom-text">You are invited to participate in a pilot study designed to explore perceptions of linguistic style in written text. Before you decide to participate, it is important that you understand why this study is being conducted and what your participation involves. Please read the following information carefully.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="header-large">Description of the Research Study</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="custom-text">In this study, we aim to investigate how readers perceive the style of written data as masculine, feminine, or gender-neutral. As an annotator, your task will involve evaluating a series of short texts based on their linguistic style, ranging from "Very Feminine" to "Very Masculine." This evaluation will focus on stylistic elements such as tone, word choice, and sentence structure rather than the content or topic of the text. Your contributions will help us create a dataset with gendered stylistic attributes, providing a foundation for understanding how people perceive gendered writing styles and the extent to which these perceptions align.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="custom-text">The findings of this study will contribute to scientific knowledge and may be included in academic publications.</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="header-large">Risks and Benefits</p>
        <p class="custom-text">The risks associated with this pilot study are minimal and comparable to those encountered during routine computer-based tasks, such as mild fatigue or boredom. Texts included in this study are written by users on blog websites and social media platforms, and may occasionally include words that could be sensitive or uncomfortable, though no extreme or offensive material is intentionally included. The data included in this study are not authored by the researchers and do not necessarily reflect their views.</p>
        <p class="custom-text">The primary benefit of participation is contributing to understanding in the field of language and perceived gender expression.</p>

        <p class="header-large">Time required</p>
        <p class="custom-text">Your participation will take an estimated 30 minutes. The time required may vary on an individual basis.</p>

        <p class="header-large">Voluntary Participation</p>
        <p class="custom-text">Participation in this study is entirely voluntary. You may choose not to participate or withdraw from the study at any point without explanation. If you decide to withdraw, your data will not be included in the analysis, and you will not be paid.</p>

        <p class="header-large">Confidentiality</p>
        <p class="custom-text">Your responses will remain completely anonymous. Please refrain from sharing any personally identifiable information during the study. The researchers will take all necessary steps to ensure the confidentiality of your contributions.</p>

        <p class="header-large">Contact</p>
        <p class="custom-text">For questions about the study or to report any adverse effects, please contact the researcher at hongyu.chen@iris.uni-stuttgart.de / Hongyu.Chen@ims.uni-stuttgart.de.</p>

        <p class="header-large">Consent</p>
        <p class="custom-text">Please indicate the information below that you are at least 18 years old, have read and understood this consent form, are comfortable using English to complete the task, and agree to participate in this research study</p>
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

    st.markdown(
        '<div class="custom-label">If you give your consent to take part, please click "I agree" below.</div>',
        unsafe_allow_html=True,
    )

    if "consent" not in st.session_state:
        st.session_state["consent"] = None

    consent_options = ["I agree", "I do not agree"]
    current_consent = st.session_state.get("consent")
    consent_index = consent_options.index(current_consent) if current_consent in consent_options else None

    st.session_state["consent"] = st.selectbox(
        "",
        options=consent_options,
        index=consent_index,
        key="consent_selectbox",
    )

    if st.session_state.get("consent") == "I agree":
        if st.button("Next"):
            st.session_state["current_page"] = "Page 2"
            st.rerun()
    elif st.session_state.get("consent") == "I do not agree":
        st.error(
            "As you do not wish to participate in this study, please return your submission on Prolific by selecting the 'Stop without completing' button."
        )
        st.button("Next", disabled=True)
    else:
        st.button("Next", disabled=True)


# ============================================================
# PAGE 2: PROLIFIC ID AND VERSION ASSIGNMENT
# ============================================================

def page2():
    st.session_state["p_id"] = st.text_input(
        "Please enter your Prolific ID",
        st.session_state.get("p_id", ""),
    )

    if st.button("Next", disabled=not st.session_state.get("p_id")):
        if st.session_state["p_id"] == ADMIN_ID:
            st.session_state["current_page"] = "Page 8"
        else:
            version = get_version_from_url()
            data = load_data(version)

            st.session_state["survey_version"] = version
            st.session_state["data"] = data
            st.session_state["responses"] = [{} for _ in range(len(data))]
            st.session_state["current_text_index"] = 0
            st.session_state["current_page"] = "Page 3"

        st.rerun()

    if st.button("Back"):
        st.session_state["current_page"] = "Page 1"
        st.rerun()


# ============================================================
# PAGE 3: GUIDELINES
# ============================================================

def page3():
    st.header("Guidelines for Annotating Masculine/Feminine Style from Texts")

    st.markdown(
        '<p class="custom-text">The goal of this study is to determine whether a text\'s style is perceived as masculine, feminine, or neutral. You will rate each text on the following scale:</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        1. **Very Feminine:** The text is strongly perceived as feminine based on linguistic style.
        2. **Somewhat Feminine:** The text has some feminine characteristics, but they are not dominant.
        3. **Neutral:** The text has no noticeable masculine or feminine characteristics.
        4. **Somewhat Masculine:** The text has some masculine characteristics, but they are not dominant.
        5. **Very Masculine:** The text is strongly perceived as masculine based on linguistic style.
        """
    )

    st.markdown(
        """
        <p class="header-large">Key Features of Feminine and Masculine Styles</p>
        <p class="custom-text">These features are general tendencies and should guide, but not constrain, your perceptions. Base your rating on the overall impression of the text.</p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="custom-bold">Feminine Style Tendencies</p>
        <div class="custom-bullet">
            <ul>
                <li><strong>Emotional Expression:</strong> Focus on feelings, relationships, empathy.</li>
                <li><strong>Collaborative Tone:</strong> Use of inclusive language and hedging.</li>
                <li><strong>Descriptive Language:</strong> Use of adjectives/adverbs and sensory details.</li>
                <li><strong>Complex Sentences:</strong> Longer sentences with subordinate clauses or narrative flow.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="custom-bold">Masculine Style Tendencies</p>
        <div class="custom-bullet">
            <ul>
                <li><strong>Fact-Focused:</strong> Emphasis on logic, data, or problem-solving.</li>
                <li><strong>Direct and Assertive:</strong> Use of authoritative statements and commands.</li>
                <li><strong>Concise Language:</strong> Short, to-the-point sentences with minimal elaboration.</li>
                <li><strong>Action-Oriented:</strong> Preference for strong verbs and goal-driven language.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="custom-bold">Neutral Style</p>
        <div class="custom-bullet">
            <ul>
                <li>The text exhibits no clear tendencies toward either feminine or masculine linguistic features.</li>
            </ul>
        </div>
        <p class="custom-bold">On the next page, you'll find examples showing how data are rated in each style for this study.</p>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Next"):
        st.session_state["current_page"] = "Page 4"
        st.rerun()

    if st.button("Back"):
        st.session_state["current_page"] = "Page 2"
        st.rerun()


# ============================================================
# PAGE 4: EXAMPLES
# ============================================================

def page4():
    st.markdown('<p class="header-large">Examples</p>', unsafe_allow_html=True)

    def display_example(example_text, scale_index, confidence_level, reasoning_text):
        st.markdown(
            f"<span style='font-size: 18px;'><b>Example:</b> {example_text}</span>",
            unsafe_allow_html=True,
        )

        st.segmented_control(
            "Select a scale:",
            STYLE_OPTIONS,
            default=scale_index,
            key=f"segmented_control_{example_text}",
        )

        st.write(f"Selected value: {scale_index}")

        st.selectbox(
            "Confidence Level",
            options=CONFIDENCE_OPTIONS,
            index=confidence_level - 1,
            key=f"confidence_{example_text}",
        )

        st.text_area("Reasoning", reasoning_text, key=f"reasoning_{example_text}")
        st.write("---")

    display_example(
        "**Text 1** I couldn’t stop thinking about how kind and thoughtful her gesture was. It felt like a warm hug on a cold day, something I really needed. Perhaps it’s silly to be so sentimental, but it meant the world to me.",
        "1: Very Feminine",
        4,
        "Emotional tone, descriptive language, and use of hedging create a strong feminine impression.",
    )

    display_example(
        "**Text 2** The atmosphere was calming, with soft lighting and gentle music in the background. It created a sense of peace and comfort that everyone seemed to enjoy.",
        "2: Somewhat Feminine",
        3,
        "Descriptive and sensory language, but less emotional depth or relational focus compared to the first example.",
    )

    display_example(
        "**Text 3** The room was brightly lit, with several tables arranged in rows. People moved around, chatting casually but focused on the tasks at hand.",
        "3: Neutral",
        3,
        "Balanced tone, straightforward description without strong emotional or action-driven language.",
    )

    display_example(
        "**Text 4** The project was completed on time due to careful planning and effective teamwork. Each task was broken down into manageable steps, ensuring efficiency throughout the process.",
        "4: Somewhat Masculine",
        2,
        "Fact-focused, concise language emphasizing planning and action.",
    )

    display_example(
        "**Text 5** The machine operates at peak efficiency under optimal conditions. Ensure all components are calibrated to specifications before proceeding with deployment.",
        "5: Very Masculine",
        4,
        "Direct, authoritative tone with technical and action-oriented language.",
    )

    st.markdown(
        '<p class="custom-bold">Our examples and reasoning are based on intuition and are provided mainly for your reference.</p>',
        unsafe_allow_html=True,
    )

    if st.button("Next"):
        st.session_state["current_page"] = "Page 5"
        st.rerun()

    if st.button("Back"):
        st.session_state["current_page"] = "Page 3"
        st.rerun()


# ============================================================
# PAGE 5: SURVEY INSTRUCTIONS
# ============================================================

def page5():
    st.header('Survey Instructions')
    st.markdown(
        """ 
        <p class="custom-text">
        There are 40 short texts provided in the following pages, which will take an estimated 25 minutes to complete. For each text (post), please provide your perception on the writing style -- masculine/feminine/neutral.
        </p>

    """,
        unsafe_allow_html=True)

    st.markdown(
        """
        <p class="custom-bold">A recap of the description to each class on the scale:</p>
        """, unsafe_allow_html=True
    )
    st.markdown("""
                    1. **Very Feminine:** The text is strongly perceived as feminine based on linguistic style.
                    2. **Somewhat Feminine:** The text has some feminine characteristics, but they are not dominant.
                    3. **Neutral:** The text has no noticeable masculine or feminine characteristics. 
                    4. **Somewhat Masculine:** The text has some masculine characteristics, but they are not dominant. 
                    5. **Very Masculine:** The text is strongly perceived as masculine based on linguistic style.
                    """)

    st.markdown(
        """
        <p class="custom-bold">Things to remember while you are annotating:</p>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="custom-bullet">
            <ul>
                <li><strong>Consider Overall Impression:</strong> Evaluate the text holistically, rather than isolating individual sentences or words.</li>
                <li><strong>Avoid Bias:</strong> Base your decision on the language used, not your assumptions about gender roles or stereotypes regarding the author who wrote the data.</li>
                <li><strong>Confidence Score:</strong> Please express your certainty/uncertantity of rating with the following confidence score:
                    <ul>
                        <li>1 = <strong>Not Confident.</strong> You were unsure or found the text ambiguous.</li>
                        <li>2 = <strong>Somewhat Confident.</strong> You made a judgment but still felt uncertain or had significant doubts. </li>
                        <li>3 = <strong>Moderately Confident.</strong> You felt reasonably sure of your judgment but had some doubts. </li>
                        <li>4 = <strong>Very Confident.</strong> You were very certain about your judgment with little to no hesitation. </li>
                    </ul> 
                </li><br>
                <li><strong>Add Comments (Optional):</strong> Briefly explain your rating if it is particularly high or low. Comments are not mandatory but help us understand your reasoning.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="custom-bold">Final Notes</p>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="custom-bullet">
            <ul> 
                <li>There is no correct answer to each rating. Please follow your intuition to make the judgement. </li>
                <li>If you’re unsure, take a moment to re-read the text and focus on its overall style.</li>
                <li>It’s okay to feel that some data are ambiguous -- please express this uncertantity with the Confidence Score.</li>
                <li>Thank you for your participation—your insights are valuable!</li>
            </ul>
        </div> <br><br>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Next"):
        st.session_state["current_page"] = "Page 6"
        st.rerun()
    if st.button("Back"):
        st.session_state["current_page"] = "Page 4"
        st.rerun()


# ============================================================
# PAGE 6: SURVEY QUESTIONS
# ============================================================

def page6():
    if "data" not in st.session_state:
        st.error("Survey data was not loaded. Please go back and enter your Prolific ID again.")
        if st.button("Back to Prolific ID"):
            st.session_state["current_page"] = "Page 2"
            st.rerun()
        return

    data = st.session_state["data"]
    st.header("Survey Questions")

    current_index = st.session_state.get("current_text_index", 0)

    if not isinstance(current_index, int):
        st.error("Invalid current index.")
        return

    try:
        row = data.iloc[current_index]
        current_text = row["short_text"]
        is_attention_check = is_attention_check_value(row.get("is_attention_check", False))
    except IndexError:
        st.error("Invalid index. The dataset does not have enough rows.")
        return

    # Display text
    if is_attention_check:
        st.markdown(
            f"""
            <div class="custom-text">
                <ul><strong>[Attention Check] {current_text}</strong></ul>
            </div><br><br>
            """,
            unsafe_allow_html=True,
        )
    else:
        regular_texts = data[data["is_attention_check"].apply(is_attention_check_value) == False]
        regular_index = regular_texts.index.get_loc(current_index)
        st.markdown(
            f"""
            <div class="custom-text">
                <ul><strong>[Text {regular_index + 1}] {current_text}</strong></ul>
            </div><br><br>
            """,
            unsafe_allow_html=True,
        )

    # Style scale
    style_key = f"style_segmented_{current_index}"

    def update_style():
        st.session_state["responses"][current_index]["style"] = st.session_state[style_key]

    current_style = st.session_state["responses"][current_index].get("style", None)

    st.segmented_control(
        "Select a scale:",
        STYLE_OPTIONS,
        default=current_style,
        key=style_key,
        on_change=update_style,
    )

    if st.session_state["responses"][current_index].get("style") is not None:
        st.write(f"Selected value: {st.session_state['responses'][current_index]['style']}")
    else:
        st.write("No value selected yet.")

    # Only regular texts have confidence and comments
    if not is_attention_check:
        current_confidence = st.session_state["responses"][current_index].get("confidence", None)
        confidence_index = CONFIDENCE_OPTIONS.index(current_confidence) if current_confidence in CONFIDENCE_OPTIONS else None

        st.session_state["responses"][current_index]["confidence"] = st.selectbox(
            "Confidence Level",
            CONFIDENCE_OPTIONS,
            index=confidence_index,
            key=f"confidence_{current_index}",
        )

        st.markdown("---")

        st.session_state["responses"][current_index]["comments"] = st.text_area(
            "Comments (Optional)",
            value=st.session_state["responses"][current_index].get("comments", ""),
            key=f"comments_{current_index}",
        )
    else:
        st.session_state["responses"][current_index]["confidence"] = ""
        st.session_state["responses"][current_index]["comments"] = ""

    # Navigation
    col1, col3 = st.columns([4, 1])

    with col1:
        if st.button("Back"):
            if current_index > 0:
                st.session_state["current_text_index"] -= 1
            else:
                st.session_state["current_page"] = "Page 5"
            st.rerun()

    with col3:
        is_style_selected = st.session_state["responses"][current_index].get("style") is not None

        if is_attention_check:
            can_go_next = is_style_selected
        else:
            is_confidence_selected = st.session_state["responses"][current_index].get("confidence") is not None
            can_go_next = is_style_selected and is_confidence_selected

        if st.button("Next", disabled=not can_go_next):
            if current_index < len(data) - 1:
                st.session_state["current_text_index"] += 1
            else:
                st.session_state["current_page"] = "Page 7"
            st.rerun()

    # Progress: count regular texts only
    attention_mask = data["is_attention_check"].apply(is_attention_check_value)
    total_regular_texts = len(data[attention_mask == False])

    completed_regular_texts = sum(
        1
        for i, response in enumerate(st.session_state["responses"])
        if not is_attention_check_value(data.iloc[i].get("is_attention_check", False))
        and response.get("style") is not None
        and response.get("confidence") not in [None, ""]
    )

    progress = completed_regular_texts / total_regular_texts if total_regular_texts > 0 else 0
    st.progress(progress)
    st.write(f"Completed {completed_regular_texts} out of {total_regular_texts} regular texts.")


# ============================================================
# PAGE 7: FEEDBACK
# ============================================================

def page7():
    st.title("Your Feedback Matters!")

    st.markdown(
        '<div class="custom-label">Thank you for participating! Please let us know if you have any questions, comments, or concerns about this survey.</div>',
        unsafe_allow_html=True,
    )

    st.session_state["feedback"] = st.text_area(
        "",
        value=st.session_state.get("feedback", ""),
    )

    if st.button("Next"):
        st.session_state["current_page"] = "Page 8"
        st.rerun()

    if st.button("Back"):
        st.session_state["current_page"] = "Page 6"
        st.rerun()


# ============================================================
# PAGE 8: SUBMISSION AND ADMIN
# ============================================================

def page8():
    st.title("End of Survey")

    if st.session_state.get("p_id") == ADMIN_ID:
        show_admin_section()
        return

    if "data" not in st.session_state:
        st.error("Survey data was not loaded. Please go back and enter your Prolific ID again.")
        return

    data = st.session_state["data"]

    st.markdown(
        """
        <div class="custom-bullet">
            <ul>
                Please complete the following two steps to record your survey response and receive your reward:
                <ul>
                    <li>1 = Click 'Submit' on this page to record your response and obtain the completion code.</li>
                    <strong>If you do not complete this step, we will not receive your data and will be unable to reward you.</strong>
                    <li>2 = Please enter the completion code on Prolific to register your submission.</li>
                </ul>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Submit", disabled=st.session_state.get("submitted", False)):
        user_id = st.session_state.get("p_id", "")

        responses_df = pd.DataFrame(st.session_state["responses"])

        # Add item metadata
        responses_df["text"] = data["short_text"]
        responses_df["is_attention_check"] = data["is_attention_check"]
        responses_df["expected_answer"] = data.get("expected_answer", "")
        responses_df["survey_version"] = st.session_state.get("survey_version", "")

        for col in ["item_id", "source_dataset", "style_condition", "reference_text"]:
            if col in data.columns:
                responses_df[col] = data[col]
            else:
                responses_df[col] = ""

        responses_df["p_id"] = user_id
        responses_df["feedback"] = st.session_state.get("feedback", "")
        responses_df["consent"] = st.session_state.get("consent", "")

        # Convert selected answers to numeric scores
        responses_df["style_score"] = responses_df["style"].str.split(":").str[0].astype(int)

        # Confidence score only exists for regular texts
        responses_df["confidence_score"] = pd.to_numeric(
            responses_df["confidence"].astype(str).str.split(":").str[0],
            errors="coerce",
        )

        # Check attention checks
        responses_df["attention_check_passed"] = ""
        attention_mask = responses_df["is_attention_check"].apply(is_attention_check_value)
        responses_df["attention_check_passed"] = pd.NA
        responses_df.loc[attention_mask, "attention_check_passed"] = (
            responses_df.loc[attention_mask, "style_score"]
            == pd.to_numeric(
                responses_df.loc[attention_mask, "expected_answer"], 
                errors="coerce")
                ).astype("boolean")

        timestamp = int(time.time())
        submission_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"survey_responses_{user_id}_{submission_time}.csv"

        try:
            responses_df.to_csv(filename, index=False)
            st.success(
                f"Thank you for your submission!\n\n"
                f"Submission code: {COMPLETION_CODE}. Please enter this code on Prolific to register your submission."
            )
            st.session_state["submitted"] = True
        except Exception as e:
            st.error(f"An error occurred while saving your response: {e}")

    elif st.button("Back"):
        st.session_state["current_page"] = "Page 7"
        st.rerun()


def show_admin_section():
    st.markdown("---")
    st.header("Admin Section")

    password = st.text_input("Enter the password to download responses", type="password")
    admin_password = os.getenv("SURVEY_ADMIN_PASSWORD", "****")

    if password == admin_password:
        st.success("Password verified. You can now download the responses.")

        files = glob.glob("survey_responses_*.csv")

        if files:
            st.write("Available response files:")
            for file in files:
                with open(file, "rb") as f:
                    st.download_button(
                        label=f"Download {file}",
                        data=f,
                        file_name=file,
                        mime="text/csv",
                    )
        else:
            st.warning("No response files found.")
    elif password:
        st.error("Incorrect password.")


# ============================================================
# MAIN APPLICATION
# ============================================================

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Page 1"

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
