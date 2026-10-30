import subprocess
import os
import datetime as dt
import pyautogui as pyau
import sounddevice as sd
import soundfile as sf

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from personality import get_ev_system_message
from memory import loadmem, savemem, summarize_history_if_needed

from rich.console import Console
from rich.panel import Panel

console = Console()

# Explicitly find and load .env in the same folder as this script
from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# 1. Connect to your active OmniRoute instance
apikey = os.getenv("OMNIROUTE_API_KEY") or "sk-omniroute"
chat = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key=apikey,
    model="auto",            # Use "auto" for smart routing, or a specific model you configured
    temperature=0.7,
    max_tokens=700
)

# PERMANENT MEMORY
memory = "permanent_memory.json"
messages = loadmem(get_ev_system_message)

exit_keywords = ["exit", "quit", "close", "bye", "goodbye"]
screenshot_key = ["clip that", "screenshot", "capture that", "ev clip that", "chat clip that"]

console.print("[bold green]OmniRoute Connected (:20128). EV text loop ready.[/bold green]")

def speak_with_piper(text):
    """Sends text to Piper TTS and plays the generated audio on Windows."""
    spoken_text = text.replace("*", "")
    output_audio = "ev_output.wav"
    piper_executable = "./piper/piper.exe"
    model_path = "./piper/glados.onnx"
    
    try:
        process = subprocess.Popen(
            [piper_executable, "--model", model_path, "--output_file", output_audio],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        process.communicate(input=text.encode("utf-8"))
        
        if os.path.exists(output_audio):
            data, fs = sf.read(output_audio)
            sd.play(data, fs)
            
    except Exception as e:
        console.print(f"[bold red][System Error: Audio generation failed - {e}][/bold red]")

while True:
    try:
        user = input("\nType your command: ").strip().lower()

        sd.stop()  # Instant cutoff
        
        if not user:
            continue
        
        if any(keyword in user for keyword in exit_keywords): 
            byebye = "Logging off. Don't let the magic smoke out."
            console.print(f"[bold yellow]{byebye}[/bold yellow]")
            speak_with_piper(byebye)
            savemem(messages)
            break
        
        if any(keyword in user for keyword in screenshot_key):
            screenshot_message = "screenshot taken"
            console.print(f"[bold yellow]{screenshot_message}[/bold yellow]")
            speak_with_piper(screenshot_message)
            timestamp = dt.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            filename = f"screenshot_{timestamp}.png"
            pyau.screenshot(filename)
            continue  
        
        # Invoke LLM through OmniRoute
        user_message = HumanMessage(content=user)
        result = chat.invoke([*messages, user_message])
        
        # Update conversational memory
        messages.append(user_message)
        messages = summarize_history_if_needed(messages, chat)
        messages.append(result)
        
        console.print(Panel(
            result.content,
            title="EV",
            subtitle="EV_Text_Mode",
            style="bold magenta",
            title_align="left",
        ))
        
        speak_with_piper(result.content)

    except Exception as loop_error: 
        console.print(f"[bold red]Loop Error: {loop_error}[/bold red]")
        speak_with_piper("Something went wrong, continuing.")
        continue