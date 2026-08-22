import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

def give_advice(emotions_list, intensity, stability, wavelength, burstiness):
    
    answers=[]
    emotions_str = ", ".join(emotions_list) if isinstance(emotions_list, list) else emotions_list

    prompt_1 = f"""
    You are an empathetic, insightful, and close friend. I will provide you with a second-by-second sequence of 60 emotion labels extracted from a 60-second video of my face, along with processed signal metrics.

    Data:
    - 60-Second Emotion Timeline: [{emotions_str}]
    - Intensity: {intensity}
    - Stability: {stability}
    - Wavelength: {wavelength}
    - Burstiness: {burstiness}

    Your task:
    Write a warm, conversational message directly to me (use "you"). Analyze the flow and shifts across the 60 emotions together with the mathematical metrics to explain what I am going through under the surface right now. Translate the emotional trajectory and math into human experiences.

    Rules:
    - Do NOT mention raw numbers, formulas, or technical terms like "wavelength" or "burstiness".
    - Speak directly to me as a perceptive, caring friend in one short, engaging paragraph.
    """

    messages = [{"role": "user", "content": prompt_1}]

    response_1 = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.7,
    )
    
    state_explanation = response_1.choices[0].message.content
    answers.append(state_explanation)
    messages.append({"role": "assistant", "content": state_explanation})
    
    prompt_2 = """
    Now, based exactly on the emotional flow and inner state you just described to me, I want you to give me a game plan for today.

    Your task:
    Write a highly actionable, friendly, and supportive message directly to me. 

    Include two key parts:
    1. What I should do today (e.g., take it easy, channel high energy into tasks, ground myself, or rest).
    2. How I should make decisions today (e.g., hold off on major choices, trust my gut, or pause before reacting).

    Keep it punchy, practical, and like a wise friend giving me a quick heads-up for the day.
    """
    
    messages.append({"role": "user", "content": prompt_2})

    response_2 = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.7,
    )

    game_plan = response_2.choices[0].message.content
    answers.append(game_plan)

    return np.array(answers)

