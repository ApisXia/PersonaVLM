import os
import json
import cv2
import base64
from typing import List, Dict, Any, Tuple
import yaml
from openai import AzureOpenAI, OpenAI
import numpy as np
from datetime import datetime
import shutil
import sys

# Import prompt loader
from prompts.prompt_loader import prompt_loader

# Interactive menu imports
try:
    import msvcrt  # Windows

    WINDOWS = True
except ImportError:
    import termios, tty, select  # Unix/Linux/Mac

    WINDOWS = False

# Load configuration
with open("configs/base_config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# ===== MODEL CONFIGURATION PRESETS =====
# Configure VLM models and tokens from config file
DECISION_MODEL = CONFIG["vlm_models"]["decision_model"]
QUESTIONNAIRE_MODEL = CONFIG["vlm_models"]["questionnaire_model"]
DECISION_MAX_TOKENS = CONFIG["vlm_models"]["decision_max_tokens"]
QUESTIONNAIRE_MAX_TOKENS = CONFIG["vlm_models"]["questionnaire_max_tokens"]
SUPPORTED_MODELS = CONFIG["vlm_models"]["supported_models"]

# Model configuration will be displayed after selections

# Model validation
if DECISION_MODEL not in SUPPORTED_MODELS:
    print(f"⚠️  Warning: Decision model '{DECISION_MODEL}' not in supported models list")
if QUESTIONNAIRE_MODEL not in SUPPORTED_MODELS:
    print(
        f"⚠️  Warning: Questionnaire model '{QUESTIONNAIRE_MODEL}' not in supported models list"
    )

# Initialize OpenAI client
client = OpenAI(
    api_key=CONFIG["openai_api_key"],
)

# Initialize Azure OpenAI client (commented out)
# client = AzureOpenAI(
#     api_key=CONFIG["azure_openai_api_key"],
#     api_version="2025-02-01-preview",
#     azure_endpoint=CONFIG["azure_openai_endpoint"],
# )


class InteractivePersonaSelector:
    """Interactive persona selector using arrow keys"""

    def __init__(self, personas_file="personas/20persona.json"):
        self.personas_file = personas_file
        self.personas = self.load_personas()
        self.persona_keys = list(self.personas.keys())
        self.selected_index = 0

    def load_personas(self) -> Dict:
        """Load personas from JSON file"""
        try:
            with open(self.personas_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Personas file '{self.personas_file}' not found!")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in personas file: {e}")
            sys.exit(1)

    def clear_screen(self):
        """Clear terminal screen"""
        os.system("cls" if os.name == "nt" else "clear")

    def get_key_input(self) -> str:
        """Get single key input (cross-platform)"""
        if WINDOWS:
            # Windows implementation
            key = msvcrt.getch()
            if key == b"\xe0":  # Special key prefix on Windows
                key = msvcrt.getch()
                if key == b"H":  # Up arrow
                    return "up"
                elif key == b"P":  # Down arrow
                    return "down"
            elif key == b"\r":  # Enter
                return "enter"
            elif key == b"\x1b":  # ESC
                return "escape"
        else:
            # Unix/Linux/Mac implementation
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1)
                if key == "\x1b":  # ESC sequence
                    key += sys.stdin.read(2)
                    if key == "\x1b[A":  # Up arrow
                        return "up"
                    elif key == "\x1b[B":  # Down arrow
                        return "down"
                    else:
                        return "escape"
                elif key == "\r" or key == "\n":  # Enter
                    return "enter"
                elif key == "\x03":  # Ctrl+C
                    return "escape"
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return "unknown"

    def truncate_text(self, text: str, max_length: int = 50) -> str:
        """Truncate text to max length with ellipsis"""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    def display_menu(self):
        """Display interactive persona selection menu"""
        self.clear_screen()
        print("┌─ Select Persona " + "─" * 50 + "┐")
        print("│ Use ↑/↓ arrows to navigate, Enter to select, ESC to exit │")
        print("├" + "─" * 64 + "┤")

        for i, persona_key in enumerate(self.persona_keys):
            persona = self.personas[persona_key]
            name = persona.get("name", persona_key)
            description = persona.get("description", "No description")

            # Truncate description for preview
            desc_preview = self.truncate_text(description.split("\n")[0], 45)

            # Format persona line
            if i == self.selected_index:
                print(
                    f"│ ► [{name}] {persona_key} ({i+1} of {len(self.persona_keys)})".ljust(
                        64
                    )
                    + "│"
                )
                print(f"│     {desc_preview}".ljust(64) + "│")
            else:
                print(f"│   [{name}] {persona_key}".ljust(64) + "│")

        print("└" + "─" * 64 + "┘")
        print(f"\nSelected: {self.persona_keys[self.selected_index]}")

    def run_interactive_selection(self) -> str:
        """Run interactive persona selection"""
        if not self.persona_keys:
            print("No personas found in file!")
            sys.exit(1)

        print("🎭 Interactive Persona Selector")
        print("=" * 50)

        # Try interactive mode
        try:
            while True:
                self.display_menu()

                key = self.get_key_input()

                if key == "up":
                    self.selected_index = (self.selected_index - 1) % len(
                        self.persona_keys
                    )
                elif key == "down":
                    self.selected_index = (self.selected_index + 1) % len(
                        self.persona_keys
                    )
                elif key == "enter":
                    selected_persona = self.persona_keys[self.selected_index]
                    self.clear_screen()
                    print(f"✅ Selected persona: {selected_persona}")
                    return selected_persona
                elif key == "escape":
                    print("\n❌ Selection cancelled.")
                    sys.exit(0)

        except KeyboardInterrupt:
            print("\n❌ Selection cancelled.")
            sys.exit(0)
        except Exception as e:
            print(f"\n⚠️  Interactive mode failed: {e}")
            return self.fallback_selection()

    def fallback_selection(self) -> str:
        """Fallback to numbered selection if interactive mode fails"""
        print("\n📝 Fallback to numbered selection:")
        print("=" * 30)

        for i, persona_key in enumerate(self.persona_keys):
            persona = self.personas[persona_key]
            name = persona.get("name", persona_key)
            print(f"{i+1:2d}. [{name}] {persona_key}")

        while True:
            try:
                choice = input(f"\nSelect persona (1-{len(self.persona_keys)}): ")
                if choice.lower() in ["q", "quit", "exit"]:
                    print("Selection cancelled.")
                    sys.exit(0)

                index = int(choice) - 1
                if 0 <= index < len(self.persona_keys):
                    selected_persona = self.persona_keys[index]
                    print(f"✅ Selected persona: {selected_persona}")
                    return selected_persona
                else:
                    print(
                        f"Please enter a number between 1 and {len(self.persona_keys)}"
                    )
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nSelection cancelled.")
                sys.exit(0)


class InteractiveVideoSelector:
    """Interactive video selector for choosing scenario from data/250722_real_sim subfolders"""

    def __init__(self, base_data_path="data/250722_real_sim"):
        self.base_data_path = base_data_path
        self.scenarios = self.load_scenarios()
        self.selected_index = 0

    def load_scenarios(self) -> List[Dict[str, str]]:
        """Load available video scenarios from subfolders"""
        scenarios = []

        if not os.path.exists(self.base_data_path):
            print(f"Warning: Base data path '{self.base_data_path}' not found!")
            return scenarios

        # Expected scenario patterns
        scenario_patterns = [
            "eye_pass",
            "eye_stop",
            "lightbar_green",
            "lightbar_red",
            "no-ehmi_pass",
            "no-ehmi_stop",
        ]

        for item in os.listdir(self.base_data_path):
            full_path = os.path.join(self.base_data_path, item)
            if os.path.isdir(full_path):
                # Check if this matches a known scenario pattern
                for pattern in scenario_patterns:
                    if pattern in item.lower():
                        # Parse eHMI type and behavior
                        if "eye" in item.lower():
                            ehmi_type = "eye"
                        elif "lightbar" in item.lower():
                            ehmi_type = "lightbar"
                        elif "no-ehmi" in item.lower():
                            ehmi_type = "no"
                        else:
                            ehmi_type = "unknown"

                        if "pass" in item.lower():
                            behavior = "pass"
                        elif "stop" in item.lower():
                            behavior = "stop"
                        elif "green" in item.lower():
                            behavior = "stop"  # green lightbar means vehicle stops
                        elif "red" in item.lower():
                            behavior = "pass"  # red lightbar means vehicle passes
                        else:
                            behavior = "unknown"

                        # Check if split subfolder exists
                        split_path = os.path.join(full_path, "split")
                        if os.path.exists(split_path):
                            scenarios.append(
                                {
                                    "name": item,
                                    "path": split_path,
                                    "ehmi_type": ehmi_type,
                                    "behavior": behavior,
                                    "description": f"eHMI: {ehmi_type}, Vehicle: {behavior}",
                                }
                            )
                        break

        # Sort scenarios by name for consistency
        scenarios.sort(key=lambda x: x["name"])
        return scenarios

    def clear_screen(self):
        """Clear terminal screen"""
        os.system("cls" if os.name == "nt" else "clear")

    def get_key_input(self) -> str:
        """Get single key input (cross-platform)"""
        if WINDOWS:
            # Windows implementation
            key = msvcrt.getch()
            if key == b"\xe0":  # Special key prefix on Windows
                key = msvcrt.getch()
                if key == b"H":  # Up arrow
                    return "up"
                elif key == b"P":  # Down arrow
                    return "down"
            elif key == b"\r":  # Enter
                return "enter"
            elif key == b"\x1b":  # ESC
                return "escape"
        else:
            # Unix/Linux/Mac implementation
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1)
                if key == "\x1b":  # ESC sequence
                    key += sys.stdin.read(2)
                    if key == "\x1b[A":  # Up arrow
                        return "up"
                    elif key == "\x1b[B":  # Down arrow
                        return "down"
                    else:
                        return "escape"
                elif key == "\r" or key == "\n":  # Enter
                    return "enter"
                elif key == "\x03":  # Ctrl+C
                    return "escape"
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return "unknown"

    def truncate_text(self, text: str, max_length: int = 40) -> str:
        """Truncate text to max length with ellipsis"""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    def display_menu(self):
        """Display interactive video scenario selection menu"""
        self.clear_screen()
        print("┌─ Select Video Scenario " + "─" * 42 + "┐")
        print("│ Use ↑/↓ arrows to navigate, Enter to select, ESC to exit │")
        print("├" + "─" * 64 + "┤")

        for i, scenario in enumerate(self.scenarios):
            name = scenario["name"]
            description = scenario["description"]

            # Truncate description for preview
            desc_preview = self.truncate_text(description, 35)

            # Format scenario line
            if i == self.selected_index:
                print(f"│ ► {name} ({i+1} of {len(self.scenarios)})".ljust(64) + "│")
                print(f"│     {desc_preview}".ljust(64) + "│")
            else:
                print(f"│   {name}".ljust(64) + "│")

        print("└" + "─" * 64 + "┘")
        print(f"\nSelected: {self.scenarios[self.selected_index]['name']}")

    def run_interactive_selection(self) -> Dict[str, str]:
        """Run interactive video scenario selection"""
        if not self.scenarios:
            print("No video scenarios found!")
            return self.fallback_selection()

        print("🎬 Interactive Video Scenario Selector")
        print("=" * 50)

        # Try interactive mode
        try:
            while True:
                self.display_menu()

                key = self.get_key_input()

                if key == "up":
                    self.selected_index = (self.selected_index - 1) % len(
                        self.scenarios
                    )
                elif key == "down":
                    self.selected_index = (self.selected_index + 1) % len(
                        self.scenarios
                    )
                elif key == "enter":
                    selected_scenario = self.scenarios[self.selected_index]
                    self.clear_screen()
                    print(f"✅ Selected scenario: {selected_scenario['name']}")
                    return selected_scenario
                elif key == "escape":
                    print("\n❌ Selection cancelled.")
                    sys.exit(0)

        except KeyboardInterrupt:
            print("\n❌ Selection cancelled.")
            sys.exit(0)
        except Exception as e:
            print(f"\n⚠️  Interactive mode failed: {e}")
            return self.fallback_selection()

    def fallback_selection(self) -> Dict[str, str]:
        """Fallback to numbered selection if interactive mode fails"""
        if not self.scenarios:
            print("No scenarios available - using default")
            return {
                "name": "eye_pass",
                "path": "data/250722_real_sim/eye_pass/split",
                "ehmi_type": "eye",
                "behavior": "pass",
                "description": "eHMI: eye, Vehicle: pass (default)",
            }

        print("\n📝 Fallback to numbered selection:")
        print("=" * 30)

        for i, scenario in enumerate(self.scenarios):
            print(f"{i+1:2d}. {scenario['name']} - {scenario['description']}")

        while True:
            try:
                choice = input(f"\nSelect scenario (1-{len(self.scenarios)}): ")
                if choice.lower() in ["q", "quit", "exit"]:
                    print("Selection cancelled.")
                    sys.exit(0)

                index = int(choice) - 1
                if 0 <= index < len(self.scenarios):
                    selected_scenario = self.scenarios[index]
                    print(f"✅ Selected scenario: {selected_scenario['name']}")
                    return selected_scenario
                else:
                    print(f"Please enter a number between 1 and {len(self.scenarios)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nSelection cancelled.")
                sys.exit(0)


class StreetCrossingPersona:
    def __init__(
        self,
        persona_type="cautious",
        personas_file="personas/20persona.json",
    ):
        # Load personas from JSON file
        with open(personas_file, "r", encoding="utf-8") as f:
            self.personas = json.load(f)

        # Get persona, raise error if it doesn't exist
        if persona_type not in self.personas:
            available_personas = list(self.personas.keys())
            raise ValueError(
                f"Persona '{persona_type}' not found in {personas_file}. Available personas: {available_personas}"
            )

        self.current_persona = self.personas[persona_type]
        self.name = self.current_persona["name"]
        self.description = self.current_persona["description"]
        self.decision_criteria = self.current_persona["decision_criteria"]


class StreetCrossingDecisionSystem:
    def __init__(
        self,
        persona_type="cautious",
        temperature=0.2,
        include_distance=False,
        video_folder="data/250722_real_sim/eye_pass/split",
        ehmi_type="no",
        personas_file="personas/20persona.json",
        output_dir=None,
        video_duration=1.0,
        max_time_steps=12,
    ):
        self.persona = StreetCrossingPersona(persona_type, personas_file)
        self.history = []
        self.current_position = 0  # Start at position 0 (safest, farthest from road)
        self.current_time = 0  # Start at time 0
        self.max_time_steps = max_time_steps
        self.positions = list(range(5))  # Positions 0-4
        self.video_duration = video_duration  # Total duration of each video in seconds
        self.all_status = []  # Track all position statuses
        self.is_crossing = False  # Track if pedestrian is crossing
        self.temperature = temperature  # VLM temperature parameter
        self.include_distance = include_distance  # Whether to include distance info
        self.video_folder = video_folder  # Path to video folder
        self.ehmi_type = ehmi_type  # eHMI type: eye, lightbar, or no

        # Create scenario-specific output folder
        if output_dir:
            self.output_folder = output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scenario_name = self.get_scenario_name()
            self.output_folder = f"out/{scenario_name}/simulation_{timestamp}"

        self.step_views_folder = os.path.join(self.output_folder, "step_views")
        os.makedirs(self.step_views_folder, exist_ok=True)
        self.used_videos = []  # Track videos used for combining

        # Detect vehicle decision and safety timeline from video folder
        self.vehicle_decision = self.detect_vehicle_decision()
        self.safety_timeline = self.get_safety_timeline()

    def get_scenario_name(self) -> str:
        """Extract scenario name from video folder path"""
        # Extract the scenario folder name (eye_pass, eye_stop, etc.)
        path_parts = self.video_folder.replace("\\", "/").split("/")

        # Look for known scenario patterns
        for part in reversed(path_parts):
            if any(
                scenario in part.lower()
                for scenario in [
                    "eye_pass",
                    "eye_stop",
                    "lightbar_green",
                    "lightbar_red",
                    "no-ehmi_pass",
                    "no-ehmi_stop",
                ]
            ):
                return part

        # Fallback - use the parent folder of split
        if "split" in path_parts:
            split_index = path_parts.index("split")
            if split_index > 0:
                return path_parts[split_index - 1]

        # Final fallback
        return "unknown_scenario"

    def detect_vehicle_decision(self) -> str:
        """Detect vehicle decision from folder name"""
        # Get the full path to check parent folders
        folder_path = self.video_folder.lower()

        if "eye_pass" in folder_path or "no-ehmi_pass" in folder_path:
            return "pass"  # vehicle will not stop, continues through
        elif "eye_stop" in folder_path or "no-ehmi_stop" in folder_path:
            return "stop"  # vehicle will stop
        elif "lightbar_green" in folder_path:
            return "stop"  # green lightbar means vehicle stops
        elif "lightbar_red" in folder_path:
            return "pass"  # red lightbar means vehicle passes (will not stop)
        elif "pass" in folder_path:
            return "pass"  # fallback for any other "pass" folders
        elif "stop" in folder_path:
            return "stop"  # fallback for any other "stop" folders
        else:
            return "unknown"

    def get_safety_timeline(self) -> List[str]:
        """Get safety timeline based on vehicle decision"""
        if self.vehicle_decision == "pass":
            # [o,o,o,o,o,x,x,x,o] - danger at time steps 5,6,7
            return [
                "safe",
                "safe",
                "safe",
                "safe",
                "safe",
                "danger",
                "danger",
                "danger",
                "safe",
            ]
        elif self.vehicle_decision == "stop":
            # [o,o,o,o,o,o,o,o,o] - all safe
            return ["safe"] * 9
        else:
            # Unknown - assume all safe
            return ["safe"] * 9

    def get_safety_status(self, time_step: int) -> str:
        """Get safety status for a specific time step"""
        if 0 <= time_step < len(self.safety_timeline):
            return self.safety_timeline[time_step]
        return "safe"

    def get_confidence_trust_evolution(self) -> str:
        """Generate confidence and trust evolution summary"""
        if not self.history:
            return "No decisions made during simulation."

        confidence_levels = [h.get("confidence", "N/A") for h in self.history]
        trust_levels = [h.get("trust", "N/A") for h in self.history]

        # Calculate trends
        valid_confidence = [c for c in confidence_levels if c != "N/A"]
        valid_trust = [t for t in trust_levels if t != "N/A"]

        summary_lines = []

        # Confidence trend
        if valid_confidence:
            initial_confidence = valid_confidence[0]
            final_confidence = valid_confidence[-1]
            avg_confidence = sum(valid_confidence) / len(valid_confidence)

            if final_confidence > initial_confidence:
                confidence_trend = "increased"
            elif final_confidence < initial_confidence:
                confidence_trend = "decreased"
            else:
                confidence_trend = "remained stable"

            summary_lines.append(
                f"Confidence: Started at {initial_confidence}/5, ended at {final_confidence}/5, averaged {avg_confidence:.1f}/5. Your confidence {confidence_trend} over time."
            )

        # Trust trend
        if valid_trust:
            initial_trust = valid_trust[0]
            final_trust = valid_trust[-1]
            avg_trust = sum(valid_trust) / len(valid_trust)

            if final_trust > initial_trust:
                trust_trend = "increased"
            elif final_trust < initial_trust:
                trust_trend = "decreased"
            else:
                trust_trend = "remained stable"

            summary_lines.append(
                f"Trust: Started at {initial_trust}/5, ended at {final_trust}/5, averaged {avg_trust:.1f}/5. Your trust in the vehicle {trust_trend} over time."
            )

        # Step-by-step progression
        step_progression = []
        for i, h in enumerate(self.history):
            conf = h.get("confidence", "N/A")
            trust = h.get("trust", "N/A")
            step_progression.append(f"Step {i+1}: Confidence {conf}/5, Trust {trust}/5")

        summary_lines.append(
            "Step-by-step progression: " + " | ".join(step_progression)
        )

        return chr(10).join(summary_lines)

    def get_safety_outcome_description(self) -> str:
        """Generate safety outcome description based on crossing behavior and vehicle decision"""
        if not self.history:
            return "No decisions were made during the simulation."

        final_decision = self.history[-1]
        final_time = final_decision["time"]
        final_position = final_decision["new_position"]

        # Check if crossed
        if self.is_crossing:
            crossing_time = final_time
            crossing_safety = self.get_safety_status(crossing_time)

            if crossing_safety == "danger":
                if self.vehicle_decision == "pass":
                    return "You initiated crossing during a dangerous time when the vehicle was passing through the intersection. This was a very risky decision that could have resulted in a serious collision."
                else:
                    return "You crossed during what appeared to be a dangerous moment, though the vehicle had stopped."
            else:
                if self.vehicle_decision == "pass":
                    return "You successfully crossed the road safely, timing your crossing to avoid the vehicle's dangerous passing period."
                else:
                    return "You crossed the road safely while the vehicle remained stopped, demonstrating good timing and caution."
        else:
            # Did not cross
            if self.vehicle_decision == "pass":
                # Check if they were in danger zone during dangerous times
                danger_times = [
                    i
                    for i, status in enumerate(self.safety_timeline)
                    if status == "danger"
                ]
                was_exposed = any(
                    h["new_position"] >= 3 and h["time"] in danger_times
                    for h in self.history
                )

                if was_exposed:
                    return "You did not complete the crossing but were exposed to danger by being close to the road during the vehicle's passing period. Your cautious decision to not cross likely prevented an accident."
                else:
                    return "You chose not to cross the road, maintaining a safe distance throughout the vehicle's passing period. This was a cautious approach that avoided all risk."
            else:
                return "You chose not to cross the road even though the vehicle had stopped and the crossing would have been safe throughout the entire period."

    def extract_frames(
        self, video_path: str, target_size: tuple = (512, 512)
    ) -> List[np.ndarray]:
        """Extract exactly 6 frames (1/4 downsample from ~24 frames)"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Extract all frames first
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, target_size)
            all_frames.append(resized_frame)

        cap.release()

        # Downsample to exactly 6 frames
        if len(all_frames) == 0:
            print(f"No frames extracted from {video_path}")
            return []

        # Calculate step size for 6 frames
        step = max(1, len(all_frames) // 6)
        selected_frames = []
        for i in range(0, min(len(all_frames), 6 * step), step):
            selected_frames.append(all_frames[i])

        # Ensure we have exactly 6 frames
        while len(selected_frames) < 6 and len(all_frames) > 0:
            selected_frames.append(all_frames[-1])
        selected_frames = selected_frames[:6]

        print(f"Extracted 6 frames from {len(all_frames)} total frames (step={step})")
        return selected_frames

    def frames_to_base64(self, frames: List[np.ndarray]) -> List[str]:
        """Convert frames to base64 strings for API"""
        base64_frames = []
        for frame in frames:
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frame = base64.b64encode(buffer).decode("utf-8")
            base64_frames.append(base64_frame)
        return base64_frames

    def get_ehmi_instruction(self) -> str:
        """Generate eHMI-specific instruction based on ehmi_type"""
        if self.ehmi_type == "eye":
            return """eHMI ATTENTION: Pay special attention to any EYE DISPLAY on vehicle screens/windshield. 
Look for eye-like graphics, animated eyes, or eye symbols on vehicle displays. 
These eye displays communicate the vehicle's awareness of pedestrians.
Include any eye display observations in your decision reasoning."""
        elif self.ehmi_type == "lightbar":
            return """eHMI ATTENTION: Pay special attention to the LIGHTBAR above the front bumper of vehicles.
Look for horizontal light strips that can display different colors (red, green, or no light).
Include any lightbar observations in your decision reasoning."""
        else:  # ehmi_type == "no"
            return ""

    def build_prompts(self, history: List[Dict]) -> Tuple[str, str]:
        """Build system and user prompts for VLM using centralized templates"""

        # Calculate distance to road based on position
        distance_map = {0: 3.2, 1: 2.4, 2: 1.6, 3: 0.8, 4: 0.0}
        current_distance = distance_map.get(self.current_position, 0.0)

        # Format decision criteria
        decision_criteria = chr(10).join(
            f"- {criterion}" for criterion in self.persona.decision_criteria
        )

        # Get system prompt from template
        system_prompt = prompt_loader.get_system_prompt(
            persona_description=self.persona.description,
            decision_criteria=decision_criteria,
        )

        # Format history using template
        history_formatted = prompt_loader.format_history(history)

        # Get user prompt from template
        user_prompt = prompt_loader.get_user_prompt(
            current_time=self.current_time,
            current_position=self.current_position,
            current_distance=current_distance,
            ehmi_instruction=self.get_ehmi_instruction(),
            history_formatted=history_formatted,
        )

        return system_prompt, user_prompt

    def get_position_status(self) -> str:
        """Generate position status display string"""
        if self.is_crossing:
            return "o-o-o-o-o-|**CROSSING**"

        status = []
        # Position 0 is farthest from road (leftmost), position 4 is closest to road (rightmost)
        for i in range(0, 5):  # 0, 1, 2, 3, 4
            if i == self.current_position:
                status.append("*")
            else:
                status.append("o")
            if i < 4:  # Add separator between positions
                status.append("-")

        # Add separator and ROAD at the end
        status.append("-|ROAD")

        return "".join(status)

    def update_position(self, decision: str) -> None:
        """Update current position based on decision"""
        if decision == "forward":
            if self.current_position == 4:
                # At position 4, forward means crossing the road
                self.is_crossing = True
            elif self.current_position < 4:
                self.current_position += 1  # Move toward road (higher position numbers)
        elif decision == "backward" and self.current_position > 0:
            self.current_position -= 1  # Move away from road (lower position numbers)
        # "stop" keeps position unchanged

    def get_next_video_path(self) -> str:
        """Determine next video path based on current position and time.
        If the exact file doesn't exist, fall back to the last available time."""
        path = f"{self.video_folder}/pos{self.current_position}_time{self.current_time}.mp4"
        if not os.path.exists(path):
            t = self.current_time - 1
            while t >= 0:
                fallback = f"{self.video_folder}/pos{self.current_position}_time{t}.mp4"
                if os.path.exists(fallback):
                    return fallback
                t -= 1
        return path

    def copy_video_to_step_views(self, video_path: str, step_number: int) -> str:
        """Copy video to step_views folder with renamed filename"""
        if os.path.exists(video_path):
            step_filename = f"step_{step_number + 1}.mp4"
            step_path = os.path.join(self.step_views_folder, step_filename)
            shutil.copy2(video_path, step_path)
            self.used_videos.append(step_path)
            return step_path
        return None

    def combine_videos(self) -> str:
        """Combine all used videos into all_agent_see.mp4"""
        if not self.used_videos:
            return None

        output_path = os.path.join(self.output_folder, "all_agent_see.mp4")

        # Get video properties from first video
        cap = cv2.VideoCapture(self.used_videos[0])
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process each video
        for video_path in self.used_videos:
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            cap.release()

        out.release()
        print(f"Combined video saved to: {output_path}")
        return output_path

    def conduct_questionnaire(self, combined_video_path: str) -> Dict[str, Any]:
        """Conduct post-simulation questionnaire using the combined video"""
        if not combined_video_path or not os.path.exists(combined_video_path):
            print("No combined video available for questionnaire")
            return {}

        # Extract 6 frames from combined video
        frames = self.extract_frames(combined_video_path)
        base64_frames = self.frames_to_base64(frames)

        # Build decision history summary with position context
        decision_summary = []
        for i, decision in enumerate(self.history):
            # Add position context explanation
            old_pos = decision["old_position"]
            new_pos = decision["new_position"]
            status = decision["status"]
            time_step = decision["time"]
            safety_status = self.get_safety_status(time_step)

            position_explanation = f"Position {old_pos} → {new_pos} (Status: {status})"
            if old_pos == 0:
                position_explanation += (
                    " - Started at safest position, farthest from road"
                )
            elif old_pos == 4 and decision["is_crossing"]:
                position_explanation += " - Reached road edge and initiated crossing"
            elif old_pos < new_pos:
                position_explanation += " - Moved closer to the road"
            elif old_pos > new_pos:
                position_explanation += " - Moved away from the road for safety"
            else:
                position_explanation += " - Stayed in same position"

            confidence_trust_info = f"Confidence: {decision.get('confidence', 'N/A')}/5 ({decision.get('confidence_reason', 'No reason')}), Trust: {decision.get('trust', 'N/A')}/5 ({decision.get('trust_reason', 'No reason')})"
            decision_summary.append(
                f"Step {i+1}: {decision['decision']} - {position_explanation} - {decision['reason']} - {confidence_trust_info}"
            )

        # Format decision criteria
        decision_criteria = chr(10).join(
            f"- {criterion}" for criterion in self.persona.decision_criteria
        )

        # Create questionnaire prompts using centralized templates
        system_prompt = prompt_loader.get_questionnaire_system(
            persona_description=self.persona.description,
            decision_criteria=decision_criteria,
        )

        user_prompt = prompt_loader.get_questionnaire(
            decision_summary=chr(10).join(decision_summary),
            confidence_trust_evolution=self.get_confidence_trust_evolution(),
            safety_outcome=self.get_safety_outcome_description(),
        )

        # Print questionnaire prompts
        print("\n=== QUESTIONNAIRE SYSTEM PROMPT ===")
        print(system_prompt)
        print("\n=== QUESTIONNAIRE USER PROMPT ===")
        print(user_prompt)
        print("=== END OF QUESTIONNAIRE PROMPTS ===\n")

        # Prepare messages for API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]

        # Add interleaved frame descriptions and images
        frame_interval = self.video_duration / 6
        for i, frame_b64 in enumerate(base64_frames):
            frame_time = i * frame_interval
            # Add frame description
            messages[1]["content"].append(
                {
                    "type": "text",
                    "text": f"Frame{i+1}: Video frame at {frame_time:.3f}s",
                }
            )
            # Add frame image
            messages[1]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}",
                        "detail": "high",
                    },
                }
            )

        # Add timing summary at the end
        messages[1]["content"].append(
            {
                "type": "text",
                "text": f"Time interval: {frame_interval:.3f}s between frames\nTotal time: {self.video_duration:.2f}s",
            }
        )

        # Call VLM API with retry
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=QUESTIONNAIRE_MODEL,
                    messages=messages,
                    max_tokens=QUESTIONNAIRE_MAX_TOKENS,
                    temperature=self.temperature,
                )
                break
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                continue

        # Parse response
        try:
            response_text = response.choices[0].message.content.strip()

            # Try to extract JSON from response
            if response_text.startswith("{") and response_text.endswith("}"):
                questionnaire_data = json.loads(response_text)
            else:
                # Look for JSON within the response
                import re

                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    questionnaire_data = json.loads(json_match.group())
                else:
                    raise json.JSONDecodeError("No JSON found", response_text, 0)

            return questionnaire_data

        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"Error parsing questionnaire response: {e}")
            return {
                "Q1": {
                    "question": "How confident are you in your decision?",
                    "selection": 3,
                    "reason": f"Unable to parse questionnaire response: {str(e)}",
                },
                "Q2": {
                    "question": "How much do you trust the autonomous vehicle?",
                    "selection": 3,
                    "reason": f"Unable to parse questionnaire response: {str(e)}",
                },
            }

    def make_decision(self, video_path: str) -> Dict[str, Any]:
        """Make crossing decision using VLM"""
        # Copy video to step_views folder
        self.copy_video_to_step_views(video_path, self.current_time)

        # Extract and process 6 frames
        frames = self.extract_frames(video_path)
        base64_frames = self.frames_to_base64(frames)

        # Build system and user prompts
        system_prompt, user_prompt = self.build_prompts(self.history)

        # Print prompts for first time step
        if self.current_time == 0:
            print("\n=== SYSTEM PROMPT FOR FIRST TIME STEP ===")
            print(system_prompt)
            print("\n=== USER PROMPT FOR FIRST TIME STEP ===")
            print(user_prompt)
            print("=== END OF PROMPTS ===\n")

        # Prepare messages for API with system and user prompts
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]

        # Add interleaved frame descriptions and images
        frame_interval = self.video_duration / 6
        for i, frame_b64 in enumerate(base64_frames):
            frame_time = i * frame_interval
            # Add frame description
            messages[1]["content"].append(
                {
                    "type": "text",
                    "text": f"Frame{i+1}: Video frame at {frame_time:.3f}s",
                }
            )
            # Add frame image
            messages[1]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}",
                        "detail": "high",
                    },
                }
            )

        # Add timing summary at the end
        messages[1]["content"].append(
            {
                "type": "text",
                "text": f"Time interval: {frame_interval:.3f}s between frames\nTotal time: {self.video_duration:.2f}s",
            }
        )

        # Call VLM API with retry
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=DECISION_MODEL,
                    messages=messages,
                    max_tokens=DECISION_MAX_TOKENS,
                    temperature=self.temperature,
                )
                break
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                continue

        # Parse response
        try:
            response_text = response.choices[0].message.content.strip()

            # Try to extract JSON from response
            if response_text.startswith("{") and response_text.endswith("}"):
                decision_data = json.loads(response_text)
            else:
                # Look for JSON within the response
                import re

                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    decision_data = json.loads(json_match.group())
                else:
                    raise json.JSONDecodeError("No JSON found", response_text, 0)

            # Validate decision format
            if "decision" not in decision_data or decision_data["decision"] not in [
                "forward",
                "stop",
                "backward",
            ]:
                raise ValueError("Invalid decision format")

            # Set default values for missing confidence/trust fields
            if "confidence" not in decision_data:
                decision_data["confidence"] = 3
            if "confidence_reason" not in decision_data:
                decision_data["confidence_reason"] = "No confidence level provided"
            if "trust" not in decision_data:
                decision_data["trust"] = 3
            if "trust_reason" not in decision_data:
                decision_data["trust_reason"] = "No trust level provided"

            return decision_data

        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"Error parsing VLM response: {e}")
            return {
                "decision": "stop",
                "reason": f"Unable to parse decision from VLM response: {str(e)}",
                "confidence": 1,
                "confidence_reason": "Error in response parsing",
                "trust": 1,
                "trust_reason": "Error in response parsing",
            }

    def run_simulation(self):
        """Run the street crossing decision simulation"""
        print("=== Street Crossing Decision Simulation V6 ===")
        print(f"Persona: {self.persona.name}")
        print(f"Description: {self.persona.description.strip()}")
        print(f"Output folder: {self.output_folder}")
        print(f"Video folder: {self.video_folder}")
        print(f"Video duration: {self.video_duration}s")
        print(f"Detected vehicle decision: {self.vehicle_decision}")
        print(f"Safety timeline: {self.safety_timeline}")
        print("=" * 50)
        print(
            f"Game Setup: 5 positions (0-4), 9 time steps (0-8), {self.video_duration}s per video"
        )
        print(f"Starting position: {self.current_position}")
        print(f"Initial status: {self.get_position_status()}")
        print("=" * 50)

        while self.current_time < self.max_time_steps and not self.is_crossing:
            print(
                f"\n--- Time Step {self.current_time} ({self.current_time * self.video_duration}s) ---"
            )
            current_video = self.get_next_video_path()
            print(f"Current position: {self.current_position}")
            print(f"Status: {self.get_position_status()}")
            print(f"Analyzing video: {current_video}")

            # Check if video exists
            if not os.path.exists(current_video):
                print(f"Video not found: {current_video}")
                break

            # Make decision
            decision = self.make_decision(current_video)

            # Display decision
            print(f"Decision: {decision['decision']}")
            print(f"Reason: {decision['reason']}")
            print(
                f"Confidence: {decision['confidence']}/5 - {decision['confidence_reason']}"
            )
            print(f"Trust: {decision['trust']}/5 - {decision['trust_reason']}")

            # Update position based on decision
            old_position = self.current_position
            old_crossing_state = self.is_crossing
            self.update_position(decision["decision"])
            new_status = self.get_position_status()
            print(f"New status: {new_status}")

            # Store status for final summary
            self.all_status.append(
                {
                    "time": self.current_time,
                    "old_position": old_position,
                    "new_position": self.current_position,
                    "decision": decision["decision"],
                    "status": new_status,
                    "is_crossing": self.is_crossing,
                }
            )

            # Add to history
            self.history.append(
                {
                    "time": self.current_time,
                    "video": current_video,
                    "old_position": old_position,
                    "new_position": self.current_position,
                    "decision": decision["decision"],
                    "reason": decision["reason"],
                    "confidence": decision["confidence"],
                    "confidence_reason": decision["confidence_reason"],
                    "trust": decision["trust"],
                    "trust_reason": decision["trust_reason"],
                    "status": new_status,
                    "is_crossing": self.is_crossing,
                }
            )

            # Check if crossing started
            if self.is_crossing and not old_crossing_state:
                print("\n🚶 PEDESTRIAN IS NOW CROSSING THE ROAD!")
                print("Simulation ends - crossing initiated.")
                break

            # Move to next time step
            self.current_time += 1

        # Combine all videos
        print("\nCombining all video clips...")
        combined_video_path = self.combine_videos()

        # Conduct post-simulation questionnaire
        print("\nConducting post-simulation questionnaire...")
        questionnaire_results = self.conduct_questionnaire(combined_video_path)

        # Save results
        self.save_results(questionnaire_results)

        # Print all status at once
        print("\n=== All Position Status Summary ===")
        for status in self.all_status:
            print(
                f"Time {status['time']}: {status['status']} (moved from {status['old_position']} to {status['new_position']} - {status['decision']})"
            )

        print(f"\n=== Simulation Complete ===")
        print(f"Output saved to: {self.output_folder}")
        print(f"Step videos saved to: {self.step_views_folder}")
        if combined_video_path:
            print(f"Combined video saved to: {combined_video_path}")
        if questionnaire_results:
            print(f"\n=== Questionnaire Results ===")
            for qid, result in questionnaire_results.items():
                print(f"{qid}: {result['question']}")
                print(f"  Selection: {result['selection']}/5")
                print(f"  Reason: {result['reason']}")
        return self.history

    def save_results(self, questionnaire_results=None):
        """Save simulation results to JSON file"""
        results = {
            "persona": {
                "name": self.persona.name,
                "description": self.persona.description,
                "criteria": self.persona.decision_criteria,
            },
            "game_setup": {
                "positions": self.positions,
                "max_time_steps": self.max_time_steps,
                "video_duration": self.video_duration,
                "starting_position": 0,
                "temperature": self.temperature,
                "include_distance": self.include_distance,
            },
            "history": self.history,
            "all_status": self.all_status,
            "summary": {
                "total_time_steps": len(self.history),
                "forward_decisions": len(
                    [h for h in self.history if h["decision"] == "forward"]
                ),
                "stop_decisions": len(
                    [h for h in self.history if h["decision"] == "stop"]
                ),
                "backward_decisions": len(
                    [h for h in self.history if h["decision"] == "backward"]
                ),
                "final_position": self.current_position,
                "completed_crossing": self.is_crossing,
            },
            "output_info": {
                "output_folder": self.output_folder,
                "step_views_folder": self.step_views_folder,
                "used_videos": [os.path.basename(v) for v in self.used_videos],
            },
            "questionnaire": questionnaire_results if questionnaire_results else {},
            "safety_analysis": {
                "vehicle_decision": self.vehicle_decision,
                "safety_timeline": self.safety_timeline,
                "safety_outcome": self.get_safety_outcome_description(),
            },
        }

        output_path = os.path.join(self.output_folder, "simulation_log.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Simulation log saved to: {output_path}")


def main():
    """Main function to run the street crossing decision system with interactive persona selection"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Street Crossing Decision Test with VLM - Real Simulation V6"
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        help="Specific persona to use (bypasses interactive selection)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
        help="Temperature parameter for VLM (0.0-2.0, lower = more deterministic)",
    )
    parser.add_argument(
        "--include-distance",
        action="store_true",
        default=True,
        help="Include distance to road information (position = meters) in VLM prompt",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default=None,
        help="Path to folder containing video files (if not specified, interactive video selector will be used)",
    )
    parser.add_argument(
        "--video-scenario",
        type=str,
        default=None,
        help="Specific video scenario to use (bypasses interactive video selection)",
    )
    parser.add_argument(
        "--ehmi-type",
        type=str,
        default="eye",
        choices=["eye", "lightbar", "no"],
        help="Type of eHMI (electronic Human-Machine Interface) to look for: eye (eye display), lightbar (front bumper lightbar), or no (no eHMI)",
    )
    parser.add_argument(
        "--personas-file",
        type=str,
        default="personas/persona_improvetransfer_v04.json",
        help="Path to JSON file containing persona definitions (default: personas/20persona.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory path (if not specified, uses default out/{scenario}/simulation_{timestamp})",
    )
    parser.add_argument(
        "--video-duration",
        type=float,
        default=1.0,
        help="Duration of each video in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    # Interactive persona selection
    if args.persona is None:
        # Use interactive selector
        selector = InteractivePersonaSelector(args.personas_file)
        selected_persona = selector.run_interactive_selection()
    else:
        # Use provided persona
        selected_persona = args.persona
        print(f"Using specified persona: {selected_persona}")

    # Interactive video scenario selection
    video_folder = args.video_folder
    ehmi_type = args.ehmi_type

    if args.video_scenario is None and args.video_folder is None:
        # Use interactive video selector
        video_selector = InteractiveVideoSelector()
        selected_scenario = video_selector.run_interactive_selection()
        video_folder = selected_scenario["path"]
        ehmi_type = selected_scenario["ehmi_type"]
        print(f"Using selected scenario: {selected_scenario['name']}")
        print(f"Video folder: {video_folder}")
        print(f"eHMI type: {ehmi_type}")
    elif args.video_scenario is not None:
        # Use specific video scenario
        video_selector = InteractiveVideoSelector()
        # Find matching scenario
        for scenario in video_selector.scenarios:
            if scenario["name"].lower() == args.video_scenario.lower():
                video_folder = scenario["path"]
                ehmi_type = scenario["ehmi_type"]
                print(f"Using specified scenario: {scenario['name']}")
                break
        else:
            print(f"Warning: Scenario '{args.video_scenario}' not found, using default")
            video_folder = args.video_folder or "data/250722_real_sim/eye_pass/split"
    else:
        # Use provided video folder
        print(f"Using specified video folder: {video_folder}")

    # Display model configuration after selections
    print(f"\n🤖 Model Configuration:")
    print(f"  Decision Model: {DECISION_MODEL} (max tokens: {DECISION_MAX_TOKENS})")
    print(
        f"  Questionnaire Model: {QUESTIONNAIRE_MODEL} (max tokens: {QUESTIONNAIRE_MAX_TOKENS})"
    )
    print(f"  Supported Models: {', '.join(SUPPORTED_MODELS)}")
    print("=" * 50)

    # Create and run system
    system = StreetCrossingDecisionSystem(
        persona_type=selected_persona,
        temperature=args.temperature,
        include_distance=args.include_distance,
        video_folder=video_folder,
        ehmi_type=ehmi_type,
        personas_file=args.personas_file,
        output_dir=args.output_dir,
        video_duration=args.video_duration,
    )
    results = system.run_simulation()

    # Print summary
    print("\n=== Final Summary ===")
    forward_count = len([r for r in results if r["decision"] == "forward"])
    stop_count = len([r for r in results if r["decision"] == "stop"])
    backward_count = len([r for r in results if r["decision"] == "backward"])

    print(f"Persona: {system.persona.name}")
    print(f"Total time steps: {len(results)}")
    print(f"Forward decisions: {forward_count}")
    print(f"Stop decisions: {stop_count}")
    print(f"Backward decisions: {backward_count}")
    print(f"Final position: {system.current_position}")


if __name__ == "__main__":
    main()
