import subprocess
import os
import datetime as dt
import pyautogui as pyau
import sounddevice as sd
import soundfile as sf

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage

from personality import get_ev_system_message
from memory import loadmem, savemem, summarize_history_if_needed

from rich.console import Console
from rich.panel import Panel

console = Console()

# 1. Initialize local LLM
chat = ChatOllama(
    model="llama3.2:3b",
    keep_alive="30m",
    num_predict=700,
    temperature=0.7 
)

# 2. Connect to local ChromaDB RAG store
DB_PATH = "./chroma_db"
retriever = None

if os.path.exists(DB_PATH):
    console.print("[dim cyan]Connecting to local RAG database...[/dim cyan]")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
else:
    console.print("[dim yellow][Warning: No ./chroma_db found. Run indexing.py to enable document search.][/dim yellow]")

# PERMANENT MEMORY
memory = "permanent_memory.json"
messages = loadmem(get_ev_system_message)

exit_keywords = ["exit", "quit", "close", "bye", "goodbye"]
screenshot_key = ["clip that", "screenshot", "capture that", "ev clip that", "chat clip that"]

console.print("[bold green]Model Loaded Successfully. EV text-only loop active.[/bold green]")

def speak_with_piper(text):
    """Sends text to Piper TTS and plays the generated audio on Windows."""
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
            # Windows-compatible audio playback
            data, fs = sf.read(output_audio)
            sd.play(data, fs)
            
            
    except Exception as e:
        console.print(f"[bold red][System Error: Audio generation failed - {e}][/bold red]")

while True:
    try:
        # Standard keyboard input
        user = input("\nType your command: ").strip().lower()

        sd.stop() #instant cutoff (NO MORE YAPPING GLADOS)
        
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
        
        # --- RAG RETRIEVAL & PROMPT INJECTION ---
        prompt_to_send = user
        
        # 1. Skip RAG lookup completely for simple greetings
        casual_inputs = ["hi", "hello", "hey", "sup", "yo", "what's up", "how are you", "good morning"]
        
        if retriever and user not in casual_inputs:
            retrieved_docs = retriever.invoke(user)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            # --- SHOW RAG HITS IN TERMINAL ---
            if context_text.strip():
                console.print(f"[dim cyan]\n[RAG Hit - Found {len(retrieved_docs)} chunk(s)]:[/dim cyan]")
                for idx, doc in enumerate(retrieved_docs, start=1):
                    source = doc.metadata.get("source", "Unknown file")
                    snippet = doc.page_content.replace("\n", " ")[:120]
                    console.print(f"[dim gray]  {idx}. [{source}] \"{snippet}...\"[/dim gray]")
                
                # 2. Guardrail prompt so she doesn't use the text unless it actually relates
                prompt_to_send = (
                    f"Context from my local files:\n{context_text}\n\n"
                    f"User Input: {user}\n\n"
                    f"Instruction: If the context above is completely unrelated to the user's input, "
                    f"IGNORE THE CONTEXT entirely and just chat normally as EV. "
                    f"If it is related, use it to answer."
                )
            else:
                console.print("[dim yellow][RAG Miss - No relevant document found][/dim yellow]")
        
        # Invoke Llama with the augmented prompt
        result = chat.invoke([*messages, HumanMessage(content=prompt_to_send)])
        
        # Store clean user prompt in conversational memory, not the stuffed context
        messages.append(HumanMessage(content=user))
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