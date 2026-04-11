import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime
import glob
import numpy as np


st.set_page_config(
    page_title="Pilot Study on Gendered Style Contrast",
    layout="centered",
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
        border: 1px solid var(--secondary-background-color);
        border-radius: 10px;
        padding: 18px;
        min-height: 240px;
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        font-size: 17px !important;
        line-height: 1.7;
        white-space: pre-wrap;
    }

    .block-container {
        max-width: 900px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)




@st.cache_data
def load_data(survey_version):
    df = pd.read_csv("survey/versioned_dataset.csv")
    df = df[df["survey_version"] == survey_version].reset_index(drop=True)

    required_cols = {
        "reference_text",
        "feminine_style",
        "masculine_style",
        "source_dataset",
        "item_id",
        "survey_version",
        "pair_in_version",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in versioned_dataset.csv: {missing}")

    df["pair_id"] = df.index + 1

    if "is_attention_check" not in df.columns:
        df["is_attention_check"] = False
    if "label" not in df.columns:
        df["label"] = ""
    if "data" not in df.columns:
        df["data"] = ""

    return df

SURVEY_VERSION = 2
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

if "pair_display_order" not in st.session_state:
    rng = np.random.default_rng()
    st.session_state["pair_display_order"] = []

    for _ in range(len(data)):
        if rng.random() < 0.5:
            st.session_state["pair_display_order"].append(
                {
                    "text_a_source": "feminine_style",
                    "text_b_source": "masculine_style",
                }
            )
        else:
            st.session_state["pair_display_order"].append(
                {
                    "text_a_source": "masculine_style",
                    "text_b_source": "feminine_style",
                }
            )


def page1():
    st.title("Pilot Study on Perception of Gendered Style Contrast")

    st.header("Consent Form")
    st.markdown(
        '<p class="custom-text">You are invited to participate in a pilot study designed to explore perceptions of contrasted linguistic style in written text. Before you decide to participate, it is important that you understand why this study is being conducted and what your participation involves. Please read the following information carefully.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="header-large">Description of the Research Study</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="custom-text">In this study, we aim to investigate <strong>whether two texts sound different in gendered style</strong>, especially along a feminine–masculine dimension. By collecting human judgments about stylistic contrast, we hope to identify what factors influence one text to sound more feminine/masculine than another. For each pair of short texts, you will rate how strongly the two texts differ in feminine versus masculine style, judge how similar the two texts are in meaning/content and in fluency or grammar, and indicate the relative gendered style direction of the pair. Please focus on how the texts are written: such as their tone, word choice, and sentence structure, rather than what the texts are about. This research can help support future work on style transfer and AI-based writing assistance.</p>',
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
    st.header("Guidelines for Comparing Text Pairs along Feminine–Masculine Style")

    st.markdown(
        """
        <p class="custom-text">
        In this study, you will compare pairs of short texts. Your main task is to judge how strongly the two texts differ along a feminine–masculine stylistic dimension, while also considering how similar they are in meaning and in fluency or grammatical acceptability.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="custom-bold">For each pair, you will answer the following questions:</p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="custom-bullet">
            <ul>
                <li><strong>Style contrast:</strong> How strongly are the two texts contrasted along the feminine–masculine style dimension?</li>
                <li><strong>Meaning similarity:</strong> How similar are the meanings or content of the two texts?</li>
                <li><strong>Grammar / fluency similarity:</strong> How similar are the two texts in fluency or grammatical acceptability?</li>
                <li><strong>Style Direction:</strong> Which text sounds more feminine relative to the other?</li>
                <li><strong>Follow-up Question:</strong> If one text is more feminine, does the other sound more masculine, or mainly just less feminine?</li>
                <li><strong>Confidence:</strong> How confident are you in your judgments?</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="custom-bold">1. Style contrast</p>
        <p class="custom-text">
        A pair has low contrast if the two texts sound stylistically similar. A pair has high contrast if the two texts feel far apart in tone, word choice, emotional expression, directness, or sentence structure.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="custom-bold">2. Meaning similarity</p>
        <p class="custom-text">
        This question asks how similar the two texts are in meaning or content. Even if the style is very different, the texts may still express a similar core idea. If the meaning is very similar, give a higher rating. If the texts express different ideas, give a lower rating.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="custom-bold">3. Grammar / fluency similarity</p>
        <p class="custom-text">
        This question asks whether the two texts are at a similar level of fluency or grammatical acceptability. If both texts feel similarly natural and well-formed, give a higher rating. If one text feels much less fluent or less grammatically acceptable than the other, give a lower rating.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="custom-bold">4. Style Direction</p>
        <p class="custom-text">
        You will first indicate which text sounds more feminine relative to the other. If one text is chosen, you will then answer a follow-up question about how to understand the other text: does it sound more masculine, or mainly just less feminine? This helps distinguish whether the pair reflects a stronger feminine–masculine contrast or a smaller difference within a similar style.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="custom-bold">5. Confidence</p>
        <p class="custom-text">
        Finally, you will indicate how confident you are in your judgments. Use this rating to reflect how certain or uncertain you feel about your answers.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="custom-text">
        Please base your judgments on how the texts are written: such as tone, wording, and sentence structure, rather than only on what the texts are about.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Next", key="page3_next"):
        st.session_state["current_page"] = "Page 4"
        st.rerun()

    if st.button("Back", key="page3_back"):
        st.session_state["current_page"] = "Page 2"
        st.rerun()


def page4():
    st.markdown('<p class="header-large">Example of the Survey Task</p>', unsafe_allow_html=True)

    st.markdown(
        """
        <p class="custom-text">
        Below is an example of the exact type of comparison you will make in the survey.
        The selected answers and explanations are shown only to illustrate how the task works.
        </p>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Text A")
        st.markdown(
            """
            <div class="pair-box">
            The project was completed on time, met all specifications, and achieved the required outcome efficiently.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown("#### Text B")
        st.markdown(
            """
            <div class="pair-box">
            Everyone worked so thoughtfully together, and it was really satisfying to see everything come together so smoothly in the end.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

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
    followup_options = ["More masculine", "Less feminine"]
    confidence_options = [
        "1: Not Confident",
        "2: Somewhat Confident",
        "3: Moderately Confident",
        "4: Very Confident",
    ]

    example_contrast = "5: Very strongly contrasted"
    example_content = "4: Mostly same"
    example_grammar = "5: Same"
    example_more_feminine = "Text B"
    example_followup = "More masculine"
    example_confidence = "4: Very Confident"

    st.markdown("**Style contrast**")
    st.segmented_control(
        "How strongly contrasted are these two texts in feminine vs masculine style?",
        options=contrast_options,
        default=example_contrast,
        key="example_contrast_segmented",
        disabled=True,
    )
    st.write(f"Selected value: {example_contrast}")
    st.markdown(
        """
        <p class="custom-text">
        <strong>Reasoning:</strong> The two texts feel far apart in style. Text A is concise, direct, and task-focused,
        while Text B is more expressive, relational, and descriptive. Because the stylistic distance between them is large,
        this pair would be rated as very strongly contrasted.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("**Meaning similarity**")
    st.segmented_control(
        "To what extent do the two texts express the same meaning or content?",
        options=content_options,
        default=example_content,
        key="example_content_segmented",
        disabled=True,
    )
    st.write(f"Selected value: {example_content}")
    st.markdown(
        """
        <p class="custom-text">
        <strong>Reasoning:</strong> Both texts describe a successful completion of a shared effort or task.
        Even though the wording and style differ, the core meaning is still fairly similar.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("**Grammar / fluency alignment**")
    st.segmented_control(
        "To what extent do the two texts have the same level of fluency / grammatical acceptability?",
        options=grammar_options,
        default=example_grammar,
        key="example_grammar_segmented",
        disabled=True,
    )
    st.write(f"Selected value: {example_grammar}")
    st.markdown(
        """
        <p class="custom-text">
        <strong>Reasoning:</strong> Both texts are fluent, natural, and grammatically acceptable.
        They differ in style, but not in overall readability or grammatical well-formedness.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Compared with the other text, which one sounds more feminine?**")

    st.radio(
        "Compared with the other text, which one sounds more feminine?",
        options=feminine_options,
        index=feminine_options.index(example_more_feminine),
        key="example_more_feminine_radio",
        disabled=True,
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <p class="custom-text">
        <strong>Reasoning:</strong> Text B uses more expressive, relational, and emotionally colored language,
        which makes it sound more feminine relative to Text A.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("**If one text sounds more feminine, how would you describe the other text in comparison?**")
    st.radio(
        "If one text sounds more feminine, how would you describe the other text in comparison?",
        options=followup_options,
        index=followup_options.index(example_followup),
        key="example_followup_radio",
        disabled=True,
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <p class="custom-text">
        <strong>Reasoning:</strong> In this example, Text A does not just sound less feminine than Text B; it also sounds more clearly masculine because it is more direct, efficient, and outcome-focused.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("**Confidence Level**")
    st.selectbox(
        "Confidence Level",
        options=confidence_options,
        index=confidence_options.index(example_confidence),
        key="example_confidence_select",
        disabled=True,
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <p class="custom-text">
        <strong>Reasoning:</strong> The contrast between the two texts is strong and relatively easy to identify,
        so this judgment can be made with high confidence.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="custom-text">
        This is only an example to show how the survey works. In the actual study, there are no strictly correct answers.
        Please use your own intuition when comparing each pair.
        </p>
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
        how similar they are in meaning and in grammar/fluency, and which text sounds more feminine.
        If one text is selected as more feminine, you will answer a short follow-up question about how the other text should be understood.
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
                <li>Comments are optional, but free feel to explain the reasoning for your judgements.</li>
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

    display_order = st.session_state["pair_display_order"][current_index]

    text_a_source = display_order["text_a_source"]
    text_b_source = display_order["text_b_source"]

    text_a = row[text_a_source]
    text_b = row[text_b_source]

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

    response["text_a_source"] = text_a_source
    response["text_b_source"] = text_b_source
    response["text_a_text"] = text_a
    response["text_b_text"] = text_b

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
    followup_options = ["More masculine", "Less feminine"]

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
    st.markdown("---")
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
    st.markdown("---")
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
    st.markdown("**Style Direction**")
    current_more_feminine = response.get("more_feminine")
    feminine_index = feminine_options.index(current_more_feminine) if current_more_feminine in feminine_options else None

    response["more_feminine"] = st.radio(
        "Compared with the other text, which one sounds more feminine?",
        options=feminine_options,
        index=feminine_index,
        key=f"more_feminine_{current_index}",
    )

    if response.get("more_feminine") == "About the same":
        response["other_text_target"] = ""
        response["other_text_interpretation"] = ""

    if response.get("more_feminine") == "Text A":
        response["other_text_target"] = "Text B"
        current_followup = response.get("other_text_interpretation")
        followup_index = followup_options.index(current_followup) if current_followup in followup_options else None

        response["other_text_interpretation"] = st.radio(
            "How would you describe Text B in comparison with Text A?",
            options=followup_options,
            index=followup_index,
            key=f"other_text_interpretation_{current_index}",
            help="Choose whether the other text feels more masculine, or mainly just less feminine.",
        )

    elif response.get("more_feminine") == "Text B":
        response["other_text_target"] = "Text A"
        current_followup = response.get("other_text_interpretation")
        followup_index = followup_options.index(current_followup) if current_followup in followup_options else None

        response["other_text_interpretation"] = st.radio(
            "How would you describe Text A in comparison with Text B?",
            options=followup_options,
            index=followup_index,
            key=f"other_text_interpretation_{current_index}",
            help="Choose whether the other text feels more masculine, or mainly just less feminine.",
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
        followup_complete = True
        if response.get("more_feminine") in ["Text A", "Text B"]:
            followup_complete = response.get("other_text_interpretation") is not None and response.get("other_text_interpretation") != ""

        required_complete = (
            response.get("contrast") is not None
            and response.get("content_alignment") is not None
            and response.get("grammar_alignment") is not None
            and response.get("more_feminine") is not None
            and followup_complete
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
        and (
            r.get("more_feminine") == "About the same"
            or (r.get("other_text_interpretation") is not None and r.get("other_text_interpretation") != "")
        )
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
        """
    )

    if st.button("Submit", key="page8_submit", disabled=st.session_state.get("submitted", False)):
        user_id = f"{st.session_state.get('p_id', '')}"

        if user_id in st.session_state.get("submitted_users", set()):
            st.warning("You have already submitted the form.")
        else:
            responses_df = pd.DataFrame(st.session_state["responses"])

            responses_df["text_a_source"] = [r.get("text_a_source", "") for r in st.session_state["responses"]]
            responses_df["text_b_source"] = [r.get("text_b_source", "") for r in st.session_state["responses"]]
            responses_df["text_a_text"] = [r.get("text_a_text", "") for r in st.session_state["responses"]]
            responses_df["text_b_text"] = [r.get("text_b_text", "") for r in st.session_state["responses"]]
            responses_df["pair_id"] = data["pair_id"]
            responses_df["feminine_style"] = data["feminine_style"]
            responses_df["masculine_style"] = data["masculine_style"]
          #  responses_df["is_attention_check"] = data["is_attention_check"]
          #  responses_df["label"] = data["label"]
          #  responses_df["data"] = data["data"]
            responses_df["p_id"] = st.session_state.get("p_id", "")
            responses_df["feedback"] = st.session_state.get("feedback", "")
            responses_df["consent"] = st.session_state.get("consent", "")
            responses_df["source_dataset"] = data["source_dataset"] 

            if "contrast" in responses_df.columns:
                responses_df["contrast_score"] = responses_df["contrast"].astype(str).str.split(":").str[0]
            else:
                responses_df["contrast_score"] = ""

            if "content_alignment" in responses_df.columns:
                responses_df["content_score"] = responses_df["content_alignment"].astype(str).str.split(":").str[0]
            else:
                responses_df["content_score"] = ""

            if "grammar_alignment" in responses_df.columns:
                responses_df["grammar_score"] = responses_df["grammar_alignment"].astype(str).str.split(":").str[0]
            else:
                responses_df["grammar_score"] = ""

            if "confidence" in responses_df.columns:
                responses_df["confidence_score"] = responses_df["confidence"].astype(str).str.split(":").str[0]
            else:
                responses_df["confidence_score"] = ""

            timestamp = int(time.time())
            submission_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"survey_responses_{user_id}_{submission_time}.csv"

            try:
                responses_df.to_csv(filename, index=False)
                st.success("Thank you for your submission!")
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