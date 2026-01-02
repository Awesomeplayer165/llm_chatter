import sys
import json
import requests
import threading
import time
import re
from datetime import datetime
import markdown
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTextEdit, QPushButton, QComboBox, 
                           QLabel, QSplitter, QGroupBox, QSpinBox, QFrame,
                           QScrollArea, QTextBrowser)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QFont, QTextCursor, QColor, QPalette, QTextCharFormat

class OllamaAPI:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    def get_models(self):
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []
    
    def chat_stream(self, model, messages, system_prompt="", temperature=0.7):
        """Send streaming chat request to Ollama"""
        try:
            # Prepare messages with system prompt
            chat_messages = []
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            chat_messages.extend(messages)
            
            payload = {
                "model": model,
                "messages": chat_messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1  # Penalize repetition
                }
            }
            
            response = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True)
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'message' in chunk and 'content' in chunk['message']:
                                yield chunk['message']['content']
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Error: {response.status_code}"
        except Exception as e:
            yield f"Error: {str(e)}"

class ThinkSection:
    def __init__(self):
        self.content = ""
        self.start_time = None
        self.end_time = None
        self.is_complete = False
    
    def add_content(self, text):
        if self.start_time is None:
            self.start_time = time.time()
        self.content += text
    
    def complete(self):
        self.end_time = time.time()
        self.is_complete = True
    
    def get_duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

class StreamingChatWorker(QThread):
    message_chunk = pyqtSignal(str, str)  # sender, chunk
    message_complete = pyqtSignal(str)  # sender
    error_occurred = pyqtSignal(str)
    think_section_start = pyqtSignal(str)  # sender
    think_section_chunk = pyqtSignal(str, str)  # sender, chunk
    think_section_complete = pyqtSignal(str, float)  # sender, duration
    
    def __init__(self, ollama_api):
        super().__init__()
        self.ollama_api = ollama_api
        self.running = False
        self.left_model = ""
        self.right_model = ""
        self.left_system_prompt = ""
        self.right_system_prompt = ""
        self.conversation_history = []
        self.delay = 2  # seconds between messages
        self.temperature = 0.7
        self.max_history_length = 20  # Keep last 20 messages (10 exchanges)
        self.turn_count = 0
        
    def set_models(self, left_model, right_model):
        self.left_model = left_model
        self.right_model = right_model
    
    def set_system_prompts(self, left_prompt, right_prompt):
        self.left_system_prompt = left_prompt
        self.right_system_prompt = right_prompt
    
    def set_delay(self, delay):
        self.delay = delay
    
    def set_temperature(self, temperature):
        self.temperature = temperature
    
    def set_max_history_length(self, max_length):
        self.max_history_length = max_length
    
    def trim_history(self):
        """Trim conversation history to prevent it from growing too long"""
        if len(self.conversation_history) > self.max_history_length:
            # Keep the first message (initial) and the most recent messages
            initial = self.conversation_history[0] if self.conversation_history else None
            recent = self.conversation_history[-self.max_history_length:]
            if initial and initial not in recent:
                self.conversation_history = [initial] + recent
            else:
                self.conversation_history = recent
    
    def start_conversation(self, initial_message):
        self.conversation_history = [{"role": "user", "content": initial_message}]
        self.turn_count = 0
        self.running = True
        self.start()
    
    def stop_conversation(self):
        self.running = False
    
    def process_streaming_response(self, sender, model, system_prompt):
        """Process streaming response and handle think sections"""
        full_response = ""
        current_think = None
        in_think_section = False
        
        # Trim history before each request to prevent loops
        self.trim_history()
        
        for chunk in self.ollama_api.chat_stream(model, self.conversation_history, system_prompt, self.temperature):
            if not self.running:
                break
                
            if chunk.startswith("Error:"):
                self.error_occurred.emit(f"{sender} model error: {chunk}")
                return None
            
            full_response += chunk
            
            # Check for think section start
            if "<think>" in chunk and not in_think_section:
                in_think_section = True
                current_think = ThinkSection()
                self.think_section_start.emit(sender)
                
                # Handle case where <think> and content are in the same chunk
                think_start_idx = chunk.find("<think>")
                before_think = chunk[:think_start_idx]
                after_think = chunk[think_start_idx + 7:]  # 7 = len("<think>")
                
                if before_think:
                    self.message_chunk.emit(sender, before_think)
                
                if after_think:
                    # Check if </think> is also in this chunk
                    if "</think>" in after_think:
                        think_end_idx = after_think.find("</think>")
                        think_content = after_think[:think_end_idx]
                        after_think_end = after_think[think_end_idx + 8:]  # 8 = len("</think>")
                        
                        current_think.add_content(think_content)
                        self.think_section_chunk.emit(sender, think_content)
                        current_think.complete()
                        self.think_section_complete.emit(sender, current_think.get_duration())
                        in_think_section = False
                        current_think = None
                        
                        if after_think_end:
                            self.message_chunk.emit(sender, after_think_end)
                    else:
                        current_think.add_content(after_think)
                        self.think_section_chunk.emit(sender, after_think)
            
            # Check for think section end
            elif "</think>" in chunk and in_think_section:
                think_end_idx = chunk.find("</think>")
                think_content = chunk[:think_end_idx]
                after_think = chunk[think_end_idx + 8:]  # 8 = len("</think>")
                
                if think_content:
                    current_think.add_content(think_content)
                    self.think_section_chunk.emit(sender, think_content)
                
                current_think.complete()
                self.think_section_complete.emit(sender, current_think.get_duration())
                in_think_section = False
                current_think = None
                
                if after_think:
                    self.message_chunk.emit(sender, after_think)
            
            # Regular content
            elif in_think_section:
                current_think.add_content(chunk)
                self.think_section_chunk.emit(sender, chunk)
            else:
                self.message_chunk.emit(sender, chunk)
        
        return full_response
    
    def run(self):
        current_sender = "left"  # Start with left model
        
        while self.running:
            if current_sender == "left":
                # Left model responds
                response = self.process_streaming_response("Left", self.left_model, self.left_system_prompt)
                
                if response is None:  # Error occurred
                    break
                
                self.message_complete.emit("Left")
                self.turn_count += 1
                self.conversation_history.append({"role": "assistant", "content": response})
                self.conversation_history.append({"role": "user", "content": response})
                current_sender = "right"
                
            else:
                # Right model responds
                response = self.process_streaming_response("Right", self.right_model, self.right_system_prompt)
                
                if response is None:  # Error occurred
                    break
                
                self.message_complete.emit("Right")
                self.turn_count += 1
                # Update conversation history
                self.conversation_history.append({"role": "assistant", "content": response})
                self.conversation_history.append({"role": "user", "content": response})
                current_sender = "left"
            
            # Wait before next message
            time.sleep(self.delay)

class ChatDisplay(QTextBrowser):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 11))
        self.current_message = ""
        self.current_message_raw = ""  # Raw markdown text
        self.current_think_section = ""
        self.think_start_marker = None
        self.message_start_position = 0  # Position where current message starts
        
        # Style the display for dark mode
        self.setStyleSheet("""
            QTextBrowser {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e3e;
                border-radius: 8px;
                padding: 10px;
            }
        """)
    
    def add_message_chunk(self, chunk):
        # Initialize message start position on first chunk
        if not self.current_message_raw:
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.message_start_position = cursor.position()
        
        self.current_message_raw += chunk
        
        # Convert markdown to HTML with dark mode styling
        # Use extensions that don't require additional dependencies
        try:
            html_content = markdown.markdown(
                self.current_message_raw,
                extensions=['fenced_code', 'codehilite', 'tables', 'nl2br']
            )
        except:
            # Fallback if codehilite is not available
            html_content = markdown.markdown(
                self.current_message_raw,
                extensions=['fenced_code', 'tables', 'nl2br']
            )
        
        # Apply dark mode CSS styling
        styled_html = f"""
        <style>
            body {{
                color: #d4d4d4;
                background-color: #1e1e1e;
                font-family: Consolas, monospace;
                font-size: 11pt;
            }}
            code {{
                background-color: #2d2d2d;
                color: #ce9178;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            pre {{
                background-color: #2d2d2d;
                color: #d4d4d4;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            pre code {{
                background-color: transparent;
                padding: 0;
            }}
            a {{
                color: #569cd6;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: #4ec9b0;
            }}
            blockquote {{
                border-left: 3px solid #6a9955;
                padding-left: 10px;
                margin-left: 0;
                color: #6a9955;
            }}
            table {{
                border-collapse: collapse;
                margin: 10px 0;
            }}
            th, td {{
                border: 1px solid #3e3e3e;
                padding: 8px;
            }}
            th {{
                background-color: #2d2d2d;
            }}
        </style>
        {html_content}
        """
        
        # Replace the message area with rendered HTML
        cursor = self.textCursor()
        cursor.setPosition(self.message_start_position)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()  # Remove old content
        cursor.insertHtml(styled_html)
        
        # Update stored message (for separator later)
        self.current_message = html_content
        
        # Move cursor to end
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
    
    def start_think_section(self):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Insert collapsible think section header
        format = QTextCharFormat()
        format.setForeground(QColor("#a0a0a0"))
        format.setBackground(QColor("#2d2d2d"))
        cursor.setCharFormat(format)
        
        self.think_start_marker = cursor.position()
        cursor.insertText("\n[🤔 Thinking...]\n")
        
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
    
    def add_think_chunk(self, chunk):
        self.current_think_section += chunk
        # Don't display think content while streaming
    
    def complete_think_section(self, duration):
        if self.think_start_marker is not None:
            cursor = self.textCursor()
            
            # Replace thinking indicator with collapsed section
            cursor.setPosition(self.think_start_marker)
            cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            
            format = QTextCharFormat()
            format.setForeground(QColor("#a0a0a0"))
            format.setBackground(QColor("#2d2d2d"))
            cursor.setCharFormat(format)
            
            collapsed_text = f"\n[🤔 Thought for {duration:.1f}s - Click to expand]\n"
            cursor.insertText(collapsed_text)
            
            self.setTextCursor(cursor)
            self.ensureCursorVisible()
            
            self.current_think_section = ""
            self.think_start_marker = None
    
    def complete_message(self):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        # Set separator color for dark mode
        format = QTextCharFormat()
        format.setForeground(QColor("#6c757d"))
        cursor.setCharFormat(format)
        cursor.insertText(f"\n\n---\n\n")
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
        self.current_message = ""
        self.current_message_raw = ""
        # Update message start position for next message
        cursor.movePosition(QTextCursor.End)
        self.message_start_position = cursor.position()
    
    def clear(self):
        """Override clear to reset message tracking"""
        super().clear()
        self.current_message = ""
        self.current_message_raw = ""
        self.message_start_position = 0

class ChatWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("LLM Conversation")
        self.setGeometry(200, 200, 1400, 900)
        
        # Apply dark mode styling to the main window
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QFrame {
                background-color: #252526;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
            }
            QLabel {
                background-color: transparent;
                color: #d4d4d4;
            }
            QSplitter::handle {
                background-color: #3e3e3e;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        
        # Left model display
        left_frame = QFrame()
        left_frame.setFrameStyle(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_frame)
        
        self.left_model_label = QLabel("Left Model: Not Set")
        self.left_model_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.left_model_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.left_model_label)
        
        self.left_display = ChatDisplay("Left")
        left_layout.addWidget(self.left_display)
        
        # Right model display
        right_frame = QFrame()
        right_frame.setFrameStyle(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_frame)
        
        self.right_model_label = QLabel("Right Model: Not Set")
        self.right_model_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.right_model_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.right_model_label)
        
        self.right_display = ChatDisplay("Right")
        right_layout.addWidget(self.right_display)
        
        # Add frames to splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_frame)
        splitter.addWidget(right_frame)
        splitter.setSizes([700, 700])
        
        layout.addWidget(splitter)
    
    def set_model_names(self, left_model, right_model):
        self.left_model_label.setText(f"Left Model: {left_model}")
        self.right_model_label.setText(f"Right Model: {right_model}")
    
    def clear_displays(self):
        self.left_display.clear()
        self.right_display.clear()
    
    def handle_message_chunk(self, sender, chunk):
        if sender == "Left":
            self.left_display.add_message_chunk(chunk)
        else:
            self.right_display.add_message_chunk(chunk)
    
    def handle_message_complete(self, sender):
        if sender == "Left":
            self.left_display.complete_message()
        else:
            self.right_display.complete_message()
    
    def handle_think_start(self, sender):
        if sender == "Left":
            self.left_display.start_think_section()
        else:
            self.right_display.start_think_section()
    
    def handle_think_chunk(self, sender, chunk):
        if sender == "Left":
            self.left_display.add_think_chunk(chunk)
        else:
            self.right_display.add_think_chunk(chunk)
    
    def handle_think_complete(self, sender, duration):
        if sender == "Left":
            self.left_display.complete_think_section(duration)
        else:
            self.right_display.complete_think_section(duration)

class DualLLMChat(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ollama_api = OllamaAPI()
        self.chat_worker = StreamingChatWorker(self.ollama_api)
        self.chat_window = None
        self.setup_ui()
        self.setup_connections()
        self.load_models()
        
    def setup_ui(self):
        self.setWindowTitle("Dual LLM Chat Controller - Ollama")
        self.setGeometry(100, 100, 800, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_group = QGroupBox("Configuration")
        control_layout = QVBoxLayout(control_group)
        
        # Model selection row
        model_layout = QHBoxLayout()
        
        # Left model selection
        left_model_layout = QVBoxLayout()
        left_model_layout.addWidget(QLabel("Left Model:"))
        self.left_model_combo = QComboBox()
        left_model_layout.addWidget(self.left_model_combo)
        
        # Right model selection
        right_model_layout = QVBoxLayout()
        right_model_layout.addWidget(QLabel("Right Model:"))
        self.right_model_combo = QComboBox()
        right_model_layout.addWidget(self.right_model_combo)
        
        # Delay setting
        delay_layout = QVBoxLayout()
        delay_layout.addWidget(QLabel("Delay (seconds):"))
        self.delay_spin = QSpinBox()
        self.delay_spin.setMinimum(1)
        self.delay_spin.setMaximum(60)
        self.delay_spin.setValue(2)
        delay_layout.addWidget(self.delay_spin)
        
        # Temperature setting
        temp_layout = QVBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.temp_spin = QSpinBox()
        self.temp_spin.setMinimum(1)
        self.temp_spin.setMaximum(200)
        self.temp_spin.setValue(70)  # 0.7 * 100 for display
        self.temp_spin.setSuffix("%")
        temp_layout.addWidget(self.temp_spin)
        
        # Max history length
        history_layout = QVBoxLayout()
        history_layout.addWidget(QLabel("Max History:"))
        self.history_spin = QSpinBox()
        self.history_spin.setMinimum(4)
        self.history_spin.setMaximum(100)
        self.history_spin.setValue(20)
        history_layout.addWidget(self.history_spin)
        
        model_layout.addLayout(left_model_layout)
        model_layout.addLayout(right_model_layout)
        model_layout.addLayout(delay_layout)
        model_layout.addLayout(temp_layout)
        model_layout.addLayout(history_layout)
        
        control_layout.addLayout(model_layout)
        
        # System prompts row
        prompts_layout = QHBoxLayout()
        
        # Left system prompt
        left_prompt_layout = QVBoxLayout()
        left_prompt_layout.addWidget(QLabel("Left System Prompt:"))
        self.left_system_prompt = QTextEdit()
        self.left_system_prompt.setMaximumHeight(100)
        self.left_system_prompt.setPlainText("You are engaged in a natural conversation with another AI. Be creative, ask questions, share ideas, and avoid simply repeating what was just said. Vary your responses and keep the conversation interesting and dynamic.")
        left_prompt_layout.addWidget(self.left_system_prompt)
        
        # Right system prompt
        right_prompt_layout = QVBoxLayout()
        right_prompt_layout.addWidget(QLabel("Right System Prompt:"))
        self.right_system_prompt = QTextEdit()
        self.right_system_prompt.setMaximumHeight(100)
        self.right_system_prompt.setPlainText("You are engaged in a natural conversation with another AI. Be creative, ask questions, share ideas, and avoid simply repeating what was just said. Vary your responses and keep the conversation interesting and dynamic.")
        right_prompt_layout.addWidget(self.right_system_prompt)
        
        prompts_layout.addLayout(left_prompt_layout)
        prompts_layout.addLayout(right_prompt_layout)
        
        control_layout.addLayout(prompts_layout)
        
        # Initial message
        initial_layout = QVBoxLayout()
        initial_layout.addWidget(QLabel("Initial Message:"))
        self.initial_message = QTextEdit()
        self.initial_message.setMaximumHeight(80)
        self.initial_message.setPlainText("Hello! How are you today?")
        initial_layout.addWidget(self.initial_message)
        
        control_layout.addLayout(initial_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Models")
        self.open_chat_button = QPushButton("Open Chat Window")
        self.start_button = QPushButton("Start Conversation")
        self.stop_button = QPushButton("Stop Conversation")
        self.clear_button = QPushButton("Clear Chat")
        
        self.stop_button.setEnabled(False)
        
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.open_chat_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.clear_button)
        
        control_layout.addLayout(button_layout)
        
        main_layout.addWidget(control_group)
        
        # Status
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
    
    def setup_connections(self):
        self.refresh_button.clicked.connect(self.load_models)
        self.open_chat_button.clicked.connect(self.open_chat_window)
        self.start_button.clicked.connect(self.start_conversation)
        self.stop_button.clicked.connect(self.stop_conversation)
        self.clear_button.clicked.connect(self.clear_chat)
        
        self.chat_worker.message_chunk.connect(self.handle_message_chunk)
        self.chat_worker.message_complete.connect(self.handle_message_complete)
        self.chat_worker.error_occurred.connect(self.display_error)
        self.chat_worker.finished.connect(self.conversation_finished)
        self.chat_worker.think_section_start.connect(self.handle_think_start)
        self.chat_worker.think_section_chunk.connect(self.handle_think_chunk)
        self.chat_worker.think_section_complete.connect(self.handle_think_complete)
        
        self.delay_spin.valueChanged.connect(lambda v: self.chat_worker.set_delay(v))
        self.temp_spin.valueChanged.connect(lambda v: self.chat_worker.set_temperature(v / 100.0))
        self.history_spin.valueChanged.connect(lambda v: self.chat_worker.set_max_history_length(v))
    
    def load_models(self):
        self.status_label.setText("Loading models...")
        models = self.ollama_api.get_models()
        
        self.left_model_combo.clear()
        self.right_model_combo.clear()
        
        if models:
            self.left_model_combo.addItems(models)
            self.right_model_combo.addItems(models)
            self.status_label.setText(f"Loaded {len(models)} models")
        else:
            self.status_label.setText("No models found. Make sure Ollama is running.")
    
    def open_chat_window(self):
        if self.chat_window is None:
            self.chat_window = ChatWindow(self)
        
        self.chat_window.show()
        self.chat_window.raise_()
        self.chat_window.activateWindow()
    
    def start_conversation(self):
        left_model = self.left_model_combo.currentText()
        right_model = self.right_model_combo.currentText()
        
        if not left_model or not right_model:
            self.status_label.setText("Please select models for both sides")
            return
        
        # Open chat window if not already open
        if self.chat_window is None:
            self.open_chat_window()
        
        # Configure worker
        self.chat_worker.set_models(left_model, right_model)
        self.chat_worker.set_system_prompts(
            self.left_system_prompt.toPlainText(),
            self.right_system_prompt.toPlainText()
        )
        self.chat_worker.set_temperature(self.temp_spin.value() / 100.0)
        self.chat_worker.set_max_history_length(self.history_spin.value())
        
        # Update chat window
        self.chat_window.set_model_names(left_model, right_model)
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText(f"Starting conversation: {left_model} ↔ {right_model}")
        
        # Start conversation
        initial_msg = self.initial_message.toPlainText()
        self.chat_worker.start_conversation(initial_msg)
    
    def stop_conversation(self):
        self.chat_worker.stop_conversation()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Conversation stopped")
    
    def conversation_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Conversation ended")
    
    def clear_chat(self):
        if self.chat_window:
            self.chat_window.clear_displays()
        self.status_label.setText("Chat cleared")
    
    def handle_message_chunk(self, sender, chunk):
        if self.chat_window:
            self.chat_window.handle_message_chunk(sender, chunk)
    
    def handle_message_complete(self, sender):
        if self.chat_window:
            self.chat_window.handle_message_complete(sender)
    
    def handle_think_start(self, sender):
        if self.chat_window:
            self.chat_window.handle_think_start(sender)
    
    def handle_think_chunk(self, sender, chunk):
        if self.chat_window:
            self.chat_window.handle_think_chunk(sender, chunk)
    
    def handle_think_complete(self, sender, duration):
        if self.chat_window:
            self.chat_window.handle_think_complete(sender, duration)
    
    def display_error(self, error_message):
        self.status_label.setText(f"Error: {error_message}")
        self.stop_conversation()

def main():
    app = QApplication(sys.argv)
    window = DualLLMChat()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
