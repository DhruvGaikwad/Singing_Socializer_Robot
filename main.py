import subprocess
import os
import json
import pyaudio
from vosk import Model, KaldiRecognizer
import pyautogui as pyau
import datetime as dt 
from langchain_ollama import ChatOllama 
from langchain_core.messages import HumanMessage
import winsound

from personality import get_ev_system_message
from memory import loadmem, savemem, summarize_history_if_needed

from rich.console import Console #use for fancy console print
from rich.panel import Panel

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

console = Console()

chat = ChatOllama(
    model="llama3.2:3b",
    keep_alive="30m",
    num_predict = 700,
    temperature=0.7 
)

# RAG DATABASE CONNECTION
console.print("[dim cyan]Connecting to local RAG database...[/dim cyan]")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# PERMANENT MEMORY
memory = "permanent_memory.json"
messages = loadmem(get_ev_system_message)

exit_keywords = ["exit","bye", "goodbye","take five"]
screenshot_key = ["clip that", "screenshot", "capture that", "ev clip that", "chat clip that"]

# VOSK SETUP 
MODEL_PATH = "vosk_model"
if not os.path.exists(MODEL_PATH):
    console.print(f"[bold red]Error: Vosk model folder '{MODEL_PATH}' not found! Download one from alphacephei.com/vosk/models[/bold red]")
    exit(1)

old_err_init = os.open(os.devnull, os.O_WRONLY)
old_stderr_init = os.dup(2)
os.dup2(old_err_init, 2)
os.close(old_err_init)

vosk_model = Model(MODEL_PATH)

os.dup2(old_stderr_init, 2)
os.close(old_stderr_init)

console.print("[bold green]Model Loaded Successfully. EV voice-only loop active .[/bold green]")

# ALSA LOGS SUPPRESSION
def mute_stderr(): 
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    return old_stderr

def unmute_stderr(old_stderr):
    os.dup2(old_stderr, 2)
    os.close(old_stderr)

def speak_with_piper(text):
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
            winsound.PlaySound(output_audio, winsound.SND_FILENAME)
            
    except Exception as e:
        console.print(f"[bold red][System Error: Audio generation failed - {e}][/bold red]")

def listen_to_user():
    """Captures microphone input locally via PyAudio and transcribes via Vosk."""
    recognizer = KaldiRecognizer(vosk_model, 16000)
    
    old_err = mute_stderr()
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
        stream.start_stream()
        unmute_stderr(old_err)
        
        console.print("[bold yellow] Listening... Speak your command.[/bold yellow]")
        
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    console.print(f"[green]You said: \"{text}\"[/green]")
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    return text.lower()
                    
    except Exception as e:
        unmute_stderr(old_err)
        console.print(f"[bold red]Vosk Speech Recognition Error: {e}[/bold red]")
        try:
            stream.stop_stream()
            stream.close()
            p.terminate()
        except:
            pass
        return ""

while True:
    try:
        user = listen_to_user()
        
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
            continue  # Prevents passing "clip that" to Llama
                
        # 1. Retrieve the top 2 relevant chunks from your PDFs/Notes based on what the user said
        retrieved_docs = retriever.invoke(user)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # 2. Quietly combine the context and the user's question into one massive prompt
        augmented_prompt = f"Use the following context to answer the question. If the context does not contain the answer, rely on your existing knowledge.\n\nContext:\n{context_text}\n\nQuestion: {user}"
        
        # 3. Send the hidden augmented prompt to Llama 3.2
        result = chat.invoke([*messages, HumanMessage(content=augmented_prompt)])
        
        # 4. CRITICAL: Only save the original 'user' string to memory, NOT the giant context prompt!
        messages.append(HumanMessage(content=user))
        messages = summarize_history_if_needed(messages, chat)
        messages.append(result)
        
        console.print(Panel(
            result.content,
            title="EV",
            subtitle="EV_1.3",
            style="bold magenta",
            title_align="left",
        ))
        
        speak_with_piper(result.content)

    except Exception as loop_error: 
        console.print(f"[bold red]Loop Error: {loop_error}[/bold red]")
        speak_with_piper("something went wrong, continuing..")
        continue