import os
import subprocess
import datetime as dt
from pathlib import Path

import pyautogui as pyau
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from personality import get_ev_system_message
from memory import loadmem, savemem, summarize_history_if_needed

from rich.console import Console
from rich.panel import Panel

console = Console()

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

apikey = os.getenv("OMNIROUTE_API_KEY") or "sk-omniroute"
chat = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key=apikey,
    model="auto",
    temperature=0.7,
    max_tokens=250
)

memory = "permanent_memory.json"
messages = loadmem(get_ev_system_message)

exit_keywords = ["exit", "quit", "close", "bye", "goodbye", "alvida", "chalo"]
screenshot_key = ["clip that", "screenshot", "capture that", "photo lo"]

console.print("[bold green]OmniRoute Connected (:20128). Dadaji Hindi loop ready.[/bold green]")

def speak_with_piper(text):
    spoken_text = text.replace("*", "").replace("#", "").replace("`", "").strip()
    if not spoken_text:
        return

    base_dir = Path(__file__).resolve().parent
    output_audio = base_dir / "ev_output.wav"
    piper_executable = base_dir / "piper" / "piper.exe"
    model_path = base_dir / "piper" / "hi_IN-rohan-medium.onnx"
    
    try:
        output_audio.unlink(missing_ok=True)
        process = subprocess.Popen(
            [str(piper_executable), "--model", str(model_path), "--output_file", str(output_audio)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        _, stderr = process.communicate(input=spoken_text.encode("utf-8"))
        
        if process.returncode != 0:
            error_message = stderr.decode("utf-8", errors="replace").strip()
            console.print(
                f"[bold red][Piper failed with exit code {process.returncode}: "
                f"{error_message}][/bold red]"
            )
            return

        if output_audio.exists():
            data, fs = sf.read(str(output_audio))
            sd.play(data, fs)
            sd.wait()
            
    except Exception as e:
        console.print(f"[bold red][System Error: Audio generation failed - {e}][/bold red]")

while True:
    try:
        user = input("\nType your command: ").strip()

        sd.stop()
        
        if not user:
            continue
        
        user_lower = user.lower()
        if any(keyword in user_lower for keyword in exit_keywords): 
            byebye = "जीते रहो बेटा, अपना ख्याल रखना। अब हम विश्राम करते हैं।"
            console.print(f"[bold yellow]{byebye}[/bold yellow]")
            speak_with_piper(byebye)
            savemem(messages)
            break
        
        if any(keyword in user_lower for keyword in screenshot_key):
            screenshot_message = "तस्वीर खींच ली गई है।"
            console.print(f"[bold yellow]{screenshot_message}[/bold yellow]")
            speak_with_piper(screenshot_message)
            timestamp = dt.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            filename = f"screenshot_{timestamp}.png"
            pyau.screenshot(filename)
            continue  
        
        user_message = HumanMessage(content=user)
        result = chat.invoke([*messages, user_message])
        
        messages.append(user_message)
        messages = summarize_history_if_needed(messages, chat)
        messages.append(result)
        
        console.print(Panel(
            result.content,
            title="Dadaji",
            subtitle="EV_Socializer",
            style="bold yellow",
            title_align="left",
        ))
        
        speak_with_piper(result.content)

    except Exception as loop_error: 
        console.print(f"[bold red]Loop Error: {loop_error}[/bold red]")
        speak_with_piper("कोई बात नहीं बेटा, आगे बढ़ते हैं।")
        continue