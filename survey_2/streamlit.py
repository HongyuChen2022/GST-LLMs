import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime
import glob


#st.set_page_config(layout="wide")



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
        font-size: 17px !important;  /* Change the font size here */
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# Function for each page
def page1():
    st.title("Pilot Study on Masculine/Feminine/Gender-Neutral Style Perception")

    st.markdown(
        '<p class="custom-text">We appreciate your feedback! Please fill out the survey below.</p>',
        unsafe_allow_html=True)
    st.header("Consent Form")
    st.markdown(
        '<p class="custom-text">You are invited to participate in a pilot study designed to explore perceptions of linguistic style in written text. Before you decide to participate, it is important that you understand why this study is being conducted and what your participation involves. Please read the following information carefully.</p>',
        unsafe_allow_html=True)
    # Apply styles to headers and other elements

    # description of the project
    st.markdown('<p class="header-large">Description of the Research Study</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="custom-text">In this study, we aim to investigate how readers perceive the style of written data as masculine, feminine, or gender-neutral. As an annotator, your task will involve evaluating a series of short data based on their linguistic style, ranging from "Very Masculine" to "Very Feminine." This evaluation will focus on stylistic elements such as tone, word choice, and sentence structure rather than the content or topic of the text. Your contributions will help us create a dataset with gendered stylistic attributes, providing a foundation for understanding how people perceive gendered writing styles, the extent to which these perceptions align, and the reactions various styles evoke.</p>',
        unsafe_allow_html=True)
    st.markdown(
        '<p class="custom-text">The findings of this study will contribute to scientific knowledge and may be included in academic publications.</p>',
        unsafe_allow_html=True)

    st.markdown(

        """
        <p class="header-large">Risks and Benefits </p>
        <p class="custom-text">The risks associated with this pilot study are minimal and comparable to those encountered during routine computer-based tasks, such as mild fatigue or boredom. Texts included in this study are written by users on blog website and social media platforms, and may occasionally include words that could be sensitive or uncomfortable, though no extreme or offensive material is intentionally included. The data included in this study are not authored by the researchers and do not necessarily reflect their views. </p>
        <p class="custom-text">The primary benefit of participation is contributing to the understanding in the field of language and perceived gender expression. </p>

        <p class="header-large">Time required </p> 
        <p class="custom-text">Your participation will take an estimated 30 minutes. The time required may vary on an individual basis </p>

        <p class="header-large">Voluntary Participation </p> 
        <p class="custom-text">Participation in this study is entirely voluntary. You may choose not to participate or withdraw from the study at any point without explanation. If you decide to withdraw, your data will not be included in the analysis, and you will not be paid. </p>

        <p class="header-large">Confidentiality </p> 
        <p class="custom-text">Your responses will remain completely anonymous. Please refrain from sharing any personally identifiable information during the study. The researchers will take all necessary steps to ensure the confidentiality of your contributions. </p>

        <p class="header-large">Contact </p> 
        <p class="custom-text">For questions about the study or to report any adverse effects, please contact the researcher at hongyu.chen@iris.uni-stuttgart.de / Hongyu.Chen@ims.uni-stuttgart.de.  </p>

        <p class="header-large">Consent </p> 
        <p class="custom-text">Please indicate the information below that you are at least 18 years old, have read and understood this consent form, are comfortable using English to complete the task, and agree to participate in this research study </p>



        """, unsafe_allow_html=True)
    st.markdown("""
                - I am 18 years old or older.
                - I have read this consent form or had it read to me.
                - My mother tongue is English.
                - I agree to participate in this research study and wish to proceed with the annotation task.
                """)
    st.markdown(
        """
        <style>
        .custom-label {
            font-size: 17px !important;  /* Change the font size here */
        }
        </style>
        <div class="custom-label">If you give your consent to take part please click 'I agree' below</div>
        """,
        unsafe_allow_html=True,
    )
    if "consent" not in st.session_state:
        st.session_state["consent"] = None

        # Get the current consent value from session state
    current_consent = st.session_state.get("consent")

    # Set the index for the selectbox
    consent_index = None if current_consent is None else ["I agree", "I do not agree"].index(current_consent)

    # Create the selectbox
    st.session_state["consent"] = st.selectbox(
        "",
        options=["I agree", "I do not agree"],
        index=consent_index,  # Set index based on the stored value
        key="consent_selectbox",  # Unique key for the selectbox
    )

    # Handle navigation based on consent
    if st.session_state.get("consent") is not None:
        if st.session_state["consent"] == "I agree":
            # Eligible participants can proceed to the next page
            if st.button("Next"):
                st.session_state["current_page"] = "Page 2"
                st.rerun()
        else:
            # Ineligible participants see a message and cannot proceed
            st.error(
                "As you do not wish to participate in this study, please return your submission on Prolific by selecting the 'Stop without completing' button."
            )
            # Disable the Next button for ineligible participants
            st.button("Next", disabled=True)
    else:
        # If no selection has been made, disable the Next button
        st.button("Next", disabled=True)
















def page2():
    st.session_state["p_id"] = st.text_input("Please enter your Prolific ID", st.session_state.get("p_id", ""))

    if st.button("Next", disabled=not st.session_state.get("p_id")):
        # Check if the entered p_id is 1 (admin)
        if st.session_state["p_id"] == "hongyuchen":
            st.session_state["current_page"] = "Page 8"  # Redirect to the admin page
        else:
            st.session_state["current_page"] = "Page 3"  # Redirect to the next page for non-admin users
        st.rerun()

    if st.button("Back"):
        st.session_state["current_page"] = "Page 1"
        st.rerun()









def page3():
    st.markdown(
        """
        <script>
            window.addEventListener('load', function() {
                window.scrollTo(0, 0);
            });
        </script>
        """,
        unsafe_allow_html=True,
    )
    st.header('Guidelines for Annotating Masculine/Feminine Style from Texts')
    st.markdown(
        """
        <p class="custom-text">The goal of this study is to determine whether a text's style is perceived as masculine, feminine, or neutral. You will rate each text on the following scale:</p>
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
        <p class="header-large">
        Key Features of Feminine and Masculine Styles
        </p>
        <p class="custom-text">These features are general tendencies and should guide, but not constrain, your perceptions. Base your rating on the overall impression of the text. 
        </p>

        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="custom-bold">Feminine Style Tendencies</p>
        <div class="custom-bullet">
            <ul>
                <li><strong>Emotional Expression:</strong> Focus on feelings, relationships, empathy (e.g., <em>I felt so overwhelmed</em>).</li>
                <li><strong>Collaborative Tone:</strong> Use of inclusive language (we, our) and hedging (maybe, perhaps).</li>
                <li><strong>Descriptive Language:</strong> Use of adjectives/adverbs and aesthetic or sensory details (e.g., <em>beautiful, softly</em>).</li>
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
                <li><strong>Fact-Focused:</strong> Emphasis on logic, data, or problem-solving (e.g., <em>The results show...</em>).</li>
                <li><strong>Direct and Assertive:</strong> Use of authoritative statements and commands (e.g., <em>This must be done</em>).</li>
                <li><strong>Concise Language:</strong> Short, to-the-point sentences with minimal elaboration.</li>
                <li><strong>Action-Oriented:</strong> Preference for strong verbs and goal-driven language (e.g., <em>achieve, complete</em>).</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """

        <p class="custom-bold">Neutral Style  </p>
        </p>
        <div class="custom-bullet">
            <ul>
                <li>The text exhibits no clear tendencies toward either feminine or masculine linguistic features. </li>
            </ul>
        </div>


        """, unsafe_allow_html=True
    )
    st.markdown(
        """

        <p class="custom-bold"> On the next page, you'll find examples showing how data are rated in each style for this study.  </p>
        </p>

        """, unsafe_allow_html=True
    )

    if st.button("Next"):
        st.session_state["current_page"] = "Page 4"
        st.rerun()
    if st.button("Back"):
        st.session_state["current_page"] = "Page 2"
        st.rerun()







def page4():
    st.markdown(
        """
        <script>
            window.addEventListener('load', function() {
                window.scrollTo(0, 0);
            });
        </script>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="header-large">
        Examples
        </p>
        """,
        unsafe_allow_html=True,
    )

    def display_example(example_text, scale_index, confidence_level, reasoning_text, font_size=""):
        # Display example text with custom font size
        st.markdown(f"<span style='font-size: {font_size};'><b>Example:</b> {example_text}</span>",
                    unsafe_allow_html=True)

        scale_options = [
            "1: Very Feminine",
            "2: Somewhat Feminine",
            "3: Neutral",
            "4: Somewhat Masculine",
            "5: Very Masculine"
        ]

        st.segmented_control(
            "Select a scale:",
            scale_options,
            default=scale_index,  # Link to session state (default to None)
            key=f"segmented_control_{example_text}",  # Unique key for the slider
        )

        # Display the selected value and its meaning
        st.write(f"Selected value: {scale_index}")

        # Confidence level in a select box
        c_options = {
            1: "1: Not Confident. You were unsure or found the text ambiguous",
            2: "2: Somewhat Confident. You made a judgment but still felt uncertain or had significant doubts",
            3: "3: Moderately Confident. You felt reasonably sure of your judgment but had some doubts",
            4: "4: Very Confident. You were very certain about your judgment with no hesitation"
        }
        st.selectbox(
            "Confidence Level",
            options=list(c_options.values()),
            index=confidence_level - 1,  # Adjust index to match confidence level
            key=f"confidence_{example_text}"
        )
        # Reasoning text box
        st.text_area("Reasoning", reasoning_text, key=f"reasoning_{example_text}")

        st.write("---")  # Separator between examples

    # Main function
    def main():

        # Example 1: Very Feminine (1)
        display_example(
            example_text="**Text 1** I couldn’t stop thinking about how kind and thoughtful her gesture was. It felt like a warm hug on a cold day, something I really needed. Perhaps it’s silly to be so sentimental, but it meant the world to me.",
            scale_index="1: Very Feminine",
            confidence_level=4,  # Very Confident
            reasoning_text="Emotional tone, descriptive language, and use of hedging (perhaps) create a strong feminine impression.",
            font_size="18px"  # Custom font size for this example
        )

        # Example 2: Somewhat Feminine (2)
        display_example(
            example_text="**Text 2** The atmosphere was calming, with soft lighting and gentle music in the background. It created a sense of peace and comfort that everyone seemed to enjoy.",
            scale_index="2: Somewhat Feminine",
            confidence_level=3,  # Moderately Confident
            reasoning_text="Descriptive and sensory language, but less emotional depth or relational focus compared to the first example.",
            font_size="18px"  # Custom font size for this example
        )

        # Example 3: Neutral (3)
        display_example(
            example_text="**Text 3** The room was brightly lit, with several tables arranged in rows. People moved around, chatting casually but focused on the tasks at hand.",
            scale_index="3: Neutral",
            confidence_level=3,  # Moderately Confident
            reasoning_text="Balanced tone, straightforward description without strong emotional or action-driven language.",
            font_size="18px"  # Custom font size for this example
        )

        # Example 4: Somewhat Masculine (4)
        display_example(
            example_text="**Text 4** The project was completed on time due to careful planning and effective teamwork. Each task was broken down into manageable steps, ensuring efficiency throughout the process.",
            scale_index="4: Somewhat Masculine",
            confidence_level=2,  # Somewhat Confident
            reasoning_text="Fact-focused, concise language emphasizing planning and action.",
            font_size="18px"  # Custom font size for this example
        )

        # Example 5: Very Masculine (5)
        display_example(
            example_text="**Text 5** The machine operates at peak efficiency under optimal conditions. Ensure all components are calibrated to specifications before proceeding with deployment.",
            scale_index="5: Very Masculine",
            confidence_level=4,  # Very Confident
            reasoning_text="Direct, authoritative tone with technical and action-oriented language.",
            font_size="18px"  # Custom font size for this example
        )
        st.markdown(
            """

            <p class="custom-bold"> Our examples and reasoning are based on intuition and are provided mainly for your reference.  </p>
            </p>

            """, unsafe_allow_html=True
        )

    if __name__ == "__main__":
        main()

    if st.button("Next"):
        st.session_state["current_page"] = "Page 5"
        st.rerun()
    if st.button("Back"):
        st.session_state["current_page"] = "Page 3"
        st.rerun()






def page5():
    st.header('Survey Instructions')
    st.markdown(
        """ 
        <p class="custom-text">
        There are 40 short data (posts) provided in the following pages, which will take an estimated 25 minutes to complete. For each text (post), please provide your perception on the writing style -- masculine/feminine/neutral.
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









# Function to display the current text and its UI elements
# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("data_pilot_attention.csv")


data = load_data()

# Initialize session state for responses
if "responses" not in st.session_state:
    st.session_state["responses"] = [{} for _ in range(len(data))]  # One dictionary per text

# Initialize session state for the current text index
if "current_text_index" not in st.session_state:
    st.session_state["current_text_index"] = 0  # Start with the first text



def page6():
    st.header("Survey Questions")

    # Get the current text index
    current_index = st.session_state["current_text_index"]

    # Ensure current_index is an integer
    if not isinstance(current_index, int):
        st.error("Invalid current_index. Please ensure it is an integer.")
        return

    # Display the current text from the dataset
    try:
        current_text = data.iloc[current_index]["short_text"]  # Replace "text_column" with the actual column name
        is_attention_check = data.iloc[current_index].get("is_attention_check", False)
    #   attention_check_instruction = data.iloc[current_index].get("attention_check_instruction", "")
    #    expected_answer = data.iloc[current_index].get("expected_answer", None)
    except IndexError:
        st.error("Invalid index. The dataset does not have enough rows.")
        return

    if is_attention_check:
        st.markdown(
            f"""
                            <div class="custom-text">
                                <ul>
                                    <strong>[Attention Check] {current_text}</strong>
                                </ul>
                            </div> <br><br>
                            """,
            unsafe_allow_html=True,
        )

    else:
        regular_texts = data[data["is_attention_check"] == False]
        regular_index = regular_texts.index.get_loc(current_index)
        st.markdown(
            f"""
                                <div class="custom-text">
                                    <ul>
                                        <strong>[Text {regular_index + 1}] {current_text}</strong>
                                    </ul>
                                </div> <br><br>
                                """,
            unsafe_allow_html=True,
        )

    def update_slider():
        st.session_state["responses"][current_index]["style"] = st.session_state["style_segmented"]

    style_options = [
        "1: Very Feminine",
        "2: Somewhat Feminine",
        "3: Neutral",
        "4: Somewhat Masculine",
        "5: Very Masculine"
    ]

    # Retrieve the slider value from session state or default to None
    slider_value = st.session_state["responses"][current_index].get("style", None)

    # Use a unique key for the slider and link it to the callback
    selected_value = st.segmented_control(
        "Select a scale:",
        style_options,
        default=slider_value,  # Link to session state (default to None)
        key="style_segmented",  # Unique key for the slider
        on_change=update_slider,  # Callback to update session state
    )

    # Display the selected value (if any)
    if st.session_state["responses"][current_index].get("style") is not None:
        selected_value = st.session_state["responses"][current_index]["style"]

        # Display the selected value and its meaning
        st.write(f"Selected value: {selected_value}")
    else:
        st.write("No value selected yet.")

    # Confidence Level Selectbox
    confidence_options = [
        "1: Not Confident. You were unsure or found the text ambiguous",
        "2: Somewhat Confident. You made a judgment but still felt uncertain or had significant doubts",
        "3: Moderately Confident. You felt reasonably sure of your judgment but had some doubts",
        "4: Very Confident. You were very certain about your judgment with no hesitation"
    ]

    # Retrieve the current confidence level from session state
    current_confidence = st.session_state["responses"][current_index].get("confidence", None)
    confidence_index = confidence_options.index(
        current_confidence) if current_confidence in confidence_options else None

    st.session_state["responses"][current_index]["confidence"] = st.selectbox(
        "Confidence Level",
        confidence_options,
        index=confidence_index,  # Set the index to the current confidence level
        key=f"confidence_{current_index}",  # Unique key for the selectbox
    )

    st.markdown("---")

    # Comments Text Area
    comments_key = f"comments_{current_index}"  # Unique key for the text area
    st.session_state["responses"][current_index]["comments"] = st.text_area(
        "Comments (Optional)",
        value=st.session_state["responses"][current_index].get("comments", ""),
        key=comments_key,
    )

    # Navigation buttons

    col1, col3 = st.columns([4, 1])
    with col1:
        if st.button("Back"):
            if current_index > 0:
                st.session_state["current_text_index"] -= 1  # Move to the previous text
            else:
                st.session_state["current_page"] = "Page 5"  # Go back to Page 5
            st.rerun()
    with col3:
        # Check if both style and confidence are selected

        is_slider_selected = st.session_state["responses"][current_index].get("style") is not None
        is_confidence_selected = st.session_state["responses"][current_index].get("confidence") is not None

        # Enable the "Next" button only if both fields are filled
        if st.button("Next", disabled=not (is_confidence_selected and is_slider_selected)):
            #   st.session_state["responses"][current_index]["confidence"] = None
            if current_index < len(data) - 1:
                st.session_state["current_text_index"] += 1  # Move to the next text
            else:
                st.session_state["current_page"] = "Page 7"  # Move to Page 7
            st.rerun()
        # if st.session_state["responses"][current_index].get("style", 0) == 0:
        #    st.warning("Please select a value between 1 and 5 to proceed.")

    # Debugging: Show all responses
    # if st.checkbox("Show all responses"):
    #   st.write(st.session_state["responses"])

    # Save responses to CSV
    # if st.button("Save Responses"):
    #   responses_df = pd.DataFrame(st.session_state["responses"])
    #  responses_df["text"] = data["data"]  # Add the text column
    # responses_df.to_csv("responses.csv", index=False)
    # st.success("Your responses are saved!")
    # Progress Slider (exclude attention checks)
    total_regular_texts = len(data[data["is_attention_check"] == False])  # Count only regular data
    completed_regular_texts = sum(
        1 for i, response in enumerate(st.session_state["responses"])
        if not data.iloc[i].get("is_attention_check", False)  # Exclude attention checks
        and response.get("style") is not None
        and response.get("confidence") is not None
    )
    progress = completed_regular_texts / total_regular_texts
    st.progress(progress)
    st.write(f"Completed {completed_regular_texts} out of {total_regular_texts} data.")








def page7():
    st.title("Your Feedback Matters!")

    st.markdown(
        """
        <style>
        .custom-label {
            font-size: 17px !important;  /* Change the font size here */
        }
        </style>
        <div class="custom-label">Thank you for participating! Please let us know if you have any questions, comments, or concerns about this survey. Your input is greatly appreciated.</div>
        """,
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





def page8():
    st.title("End of Survey")

    st.markdown(
        """
        <div class="custom-bullet">
            <ul>
                Please complete the following two steps to record your survey response and receive your reward:
                    <ul>
                        <li> 1 = Click 'Submit' on this page to record your response and to obtain the completion code </li>
                        <strong>If you do not complete the first step, we will not receive your data and will be unable to reward you.</strong>
                        <li> 2 = Please enter the completion code on Prolific to register your submission </li>
                    </ul> 
                </li><br>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )



    if st.button("Submit", disabled=st.session_state.get("submitted", False)):  # if the button can be clicked
        # Create a unique identifier for the user
        user_id = f"{st.session_state['p_id']}"

        # Check if the user has already submitted the form
        if user_id in st.session_state.get("submitted_users", set()):
            st.warning("You have already submitted the form. Thank you for your feedback!")
        else:
            # Save data
            responses_df = pd.DataFrame(st.session_state["responses"])
            responses_df["data"] = data["short_text"]  # Add the text column
            responses_df["text_id"] = data["id"]
            responses_df["label"] = data["label"]
            responses_df["data"] = data["data"]
            responses_df["p_id"] = st.session_state.get("p_id", "")  # Add Prolific ID
            responses_df["feedback"] = st.session_state.get("feedback", "")  # Add feedback
            responses_df["consent"] = st.session_state.get("consent", "")
            responses_df["style_score"] = responses_df['style'].str.split(":").str[0].astype(int)
            responses_df["confidence_score"] = responses_df['confidence'].str.split(":").str[0].astype(int)

            # Save to CSV

            # Generate a unique filename using Prolific ID and timestamp
            timestamp = int(time.time())  # Current timestamp
            submission_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
            # Human-readable format
            filename = f"survey_responses_{user_id}_{submission_time}.csv"

            try:
                # Save the responses to a new file
                responses_df.to_csv(filename, index=False)
                st.success(f"Thank you for your submission! "
                           f"\n\n Submission code: ****. Please enter this code on Prolific to register your submission")

                # Mark the form as submitted
                st.session_state["submitted"] = True

                # Add the user to the set of submitted users
                if "submitted_users" not in st.session_state:
                    st.session_state["submitted_users"] = set()
                st.session_state["submitted_users"].add(user_id)
            except Exception as e:
                st.error(f"An error occurred while saving your response: {e}")

    elif st.button("Back"):
        st.session_state["current_page"] = "Page 7"
        st.rerun()

    user_id = f"{st.session_state['p_id']}"
    if user_id == "****":
        st.markdown("---")
        st.header("Admin Section")
        password = st.text_input("Enter the password to download responses", type="password")

        admin_password = os.getenv("****", "****")

        if password == admin_password:
            st.success("Password verified. You can now download the responses.")

            # List all files matching the pattern "survey_responses_*.csv"
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
    # else:
    #   st.warning("You do not have permission to access the admin section.")








# Main Application
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

