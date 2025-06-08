"""
Define a new environment, subclassing the MultiTurnEnv over from the multiturn_env.py file,
that is used to play the NYT Connections game.
"""
from __future__ import annotations
import re
import random
from typing import List, Dict, Any, Tuple, Optional, Callable
from types import SimpleNamespace
from datasets import Dataset
from pydantic import BaseModel, Field

from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import Parser
from verifiers.rubrics import Rubric

import json
import random
import requests
from datasets import Dataset

from abc import abstractmethod
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Union

from openai import OpenAI

from verifiers.envs.environment import Environment

# Fixed configuration
RANDOM_SEED = 42
NYT_CONNECTIONS_URL = "https://raw.githubusercontent.com/Eyefyre/NYT-Connections-Answers/refs/heads/main/connections.json"

NYT_CONNECTIONS_SYSTEM_PROMPT = """\
You are playing NYT Connections, a word puzzle game where you need to find groups of 4 words that share something in common.

Rules:
- You have 16 words total arranged in 4 groups of 4 words each
- Each group has a specific theme or connection
- You have 4 lives (mistakes allowed)
- You must guess exactly 4 words at once
- If you guess correctly, that group is revealed and removed from the board
- If you guess incorrectly, you lose a life
- The game ends when you find all groups or run out of lives

For each turn:
1. Think step-by-step inside <think>...</think> tags about:
   - What connections you can see between words
   - Which group seems most confident
   - Why those 4 words belong together
2. Make your guess inside <guess>...</guess> tags with exactly 4 words separated by commas

Example format:
<think>
I can see potential connections... Let me analyze the words on the board...
I'm most confident about these 4 words because...
</think>

<guess>WORD1, WORD2, WORD3, WORD4</guess>"""


class NYTConnectionsParser(Parser):
    """Parser for NYT Connections that extracts thinking and comma-separated word guesses."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def parse(self, text: str) -> SimpleNamespace:
        """Parse thinking and guess from the text."""
        result = SimpleNamespace()
        
        # Extract thinking
        think_pattern = r"<think>\s*(.*?)\s*</think>"
        think_match = re.search(think_pattern, text, re.DOTALL)
        result.think = think_match.group(1).strip() if think_match else None
        
        # Extract guess
        guess_pattern = r"<guess>\s*(.*?)\s*</guess>"
        guess_match = re.search(guess_pattern, text, re.DOTALL)
        if guess_match:
            guess_text = guess_match.group(1).strip()
            # Parse comma-separated words and clean them
            words = [word.strip().upper() for word in guess_text.split(',')]
            # Filter out empty strings
            words = [word for word in words if word]
            result.guess = words if len(words) == 4 else None
        else:
            result.guess = None
            
        return result
    
    def get_format_reward_func(self) -> Callable:
        """Return a reward function that checks format compliance."""
        def format_reward_func(completion: List[Dict[str, str]], **kwargs) -> float:
            model_messages = self.get_assistant_messages(completion)
            if not model_messages:
                return 0.0
            
            total_score = 0.0
            for msg in model_messages:
                content = msg['content']
                parsed = self.parse(content)
                
                score = 0.0
                # Check for thinking section
                if parsed.think is not None:
                    score += 0.3
                
                # Check for properly formatted guess
                if parsed.guess is not None and len(parsed.guess) == 4:
                    score += 0.7
                elif parsed.guess is not None:
                    score += 0.3  # Partial credit for attempting guess format
                
                total_score += score
            
            return total_score / len(model_messages) if model_messages else 0.0
        
        return format_reward_func
    

class ConnectionsGroup(BaseModel):
    group: str
    members: List[str]
    level: int

    def __repr__(self) -> str:
        return f"{self.group}: {', '.join(self.members)}"

class NYTGameState(BaseModel):
    """Pydantic model for NYT Connections game state."""
    remaining_words: List[str]
    answer: List[ConnectionsGroup]
    lives: int = 4
    found_groups: List[Dict[str, Any]] = []

    @classmethod
    def initialize(cls, answer_dict: List[Dict[str, Any]]) -> NYTGameState:
        """Initialize a new game state from an answer."""
        all_words = []
        connections_groups = []
        for group in answer_dict:
            connections_group = ConnectionsGroup(**group)
            all_words.extend(connections_group.members)
            connections_groups.append(connections_group)

        # we need to shuffle the words
        random.shuffle(all_words)
        
        return cls(
            lives=4,
            found_groups=[],
            remaining_words=[word.upper() for word in all_words],
            answer=connections_groups
        )

    def get_current_prompt(self) -> str:
        """Format the current board state for display."""
        if self.found_groups:
            board_text = "SOLVED GROUPS:\n"
            for group in self.found_groups:
                board_text += f"{group}\n"
            board_text += "\nREMAINING WORDS:\n"
        else:
            board_text = "WORDS ON THE BOARD:\n"
        
        # Show remaining words in a list
        board_text += ", ".join(self.remaining_words)
        
        return board_text.strip()
    
    def check_guess(self, guess: List[str]) -> Tuple[bool, Optional[ConnectionsGroup]]:
        """Check if a guess matches any of the remaining groups."""
        guess_set = set(word.upper() for word in guess)
        for group in self.answer:
            group_set = set(word.upper() for word in group.members)
            if guess_set == group_set:
                return True, group
        
        return False, None
    
    def is_completed(self) -> bool:
        """Check if the game is completed."""
        return self.lives <= 0 or len(self.found_groups) == 4
    
    def decrease_lives(self) -> None:
        """Decrease the number of lives by 1."""
        self.lives -= 1
    
    def remove_found_words(self, group: ConnectionsGroup) -> None:
        """Remove found words from remaining words and add group to found groups."""
        for word in group.members:
            if word.upper() in self.remaining_words:
                self.remaining_words.remove(word.upper())
        random.shuffle(self.remaining_words)
        self.found_groups.append(group)

class NYTConnectionsEnv(MultiTurnEnv):
    """Environment for NYT Connections word puzzle game."""

    def __init__(self, 
                 max_turns: int = 4,
                 num_eval_samples: int = 100,
                 **kwargs):
        
        parser = NYTConnectionsParser()
        rubric = Rubric(parser=parser)
        
        # Initialize datasets if not provided
        dataset, eval_dataset = self._init_nyt_datasets(
            num_eval_samples=num_eval_samples,
        )
        
        def success_reward_func(**kwargs) -> float:
            """Reward for successfully solving the puzzle."""
            state = kwargs.get('state', {})
            found_groups = state.get('found_groups', [])
            return 1.0 if len(found_groups) == 4 else 0.0
        
        def efficiency_reward_func(**kwargs) -> float:
            """Reward based on efficiency (fewer wrong guesses)."""
            state = kwargs.get('state', {})
            lives_used = 4 - state.get('lives', 4)
            found_groups = len(state.get('found_groups', []))
            if found_groups == 4:  # Only give efficiency bonus if solved
                return max(0, (4 - lives_used) / 4)
            return 0.0
        
        def partial_progress_reward_func(**kwargs) -> float:
            """Reward for partial progress (finding some groups)."""
            state = kwargs.get('state', {})
            found_groups = len(state.get('found_groups', []))
            return found_groups / 4
        
        rubric.add_reward_func(success_reward_func, weight=1.0)
        rubric.add_reward_func(efficiency_reward_func, weight=0.3)
        rubric.add_reward_func(partial_progress_reward_func, weight=0.2)
        rubric.add_reward_func(parser.get_format_reward_func(), weight=0.1)
        
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=NYT_CONNECTIONS_SYSTEM_PROMPT,
            parser=parser,
            rubric=rubric,
            message_type='chat',
            max_turns=max_turns,
            **kwargs
        )
        self.parser = parser
        self.rubric = rubric

    
    def env_response(self, 
                     messages: List[Dict[str, Any]], 
                     state: NYTGameState, 
                     **kwargs: Any) -> Tuple[Dict[str, Any], NYTGameState]:
        """Generate environment response after a player guess."""
        
        # Parse the player's guess
        last_message = messages[-1]['content']
        parsed = self.parser.parse(last_message)
        guess = parsed.guess
        
        if guess is None or len(guess) != 4:
            response = "Please provide exactly 4 words in your guess, separated by commas."
            state.decrease_lives()
        else:
            # Check if guess is correct
            is_correct, matched_group = state.check_guess(guess)
            
            if is_correct:
                # Remove found words from remaining words
                state.remove_found_words(matched_group)
                
                if state.is_completed():
                    response = f"🎉 CORRECT! You found: {matched_group}\n\nCongratulations! You solved the puzzle!"
                else:
                    response = f"🎉 CORRECT! You found: {matched_group}\n\n{state.get_current_prompt()}"
            else:
                state.decrease_lives()
                if state.lives <= 0:
                    response = f"❌ Incorrect guess. Game over! You ran out of lives.\n\nThe correct groups were:\n"
                    for group in state.answer:
                        response += f"{group}\n"
                else:
                    response = f"❌ Incorrect guess. Lives remaining: {state.lives}\n\n{state.get_current_prompt()}"
        
        env_message = {"role": "user", "content": response}
        return env_message, state
    
    def rollout(self,
                client: OpenAI,
                model: str,
                prompt: Union[str, List[Dict[str, Any]]],
                answer: str,
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        is_completed = False
        assert isinstance(prompt, list)
        messages = deepcopy(prompt) 
        completion = []
        turn = 0

        state = NYTGameState.initialize(json.loads(answer))
        while not is_completed:
            if state.is_completed():
                is_completed = True
                break
            response = self.get_model_response(
                prompt=messages,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type=self.message_type
            )
            has_error = response.startswith("[ERROR]")
            messages.append({"role": "assistant", "content": response})
            completion.append({"role": "assistant", "content": response})
            turn += 1
            if state.is_completed() or turn >= self.max_turns or has_error:
                is_completed = True
            else:
                env_msg, state = self.env_response(messages, state, **kwargs)
                messages.append(env_msg)
                completion.append(env_msg)
        return completion, state.model_dump()
    
    def _init_nyt_datasets(self, num_eval_samples: int = 100) -> Tuple[Dataset, Dataset]:
        """
        Initialize train and eval datasets from NYT Connections JSON data.
        
        Args:
            num_samples: Number of training samples
            num_eval_samples: Number of evaluation samples
            seed: Random seed for shuffling
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        # Fetch JSON data
        response = requests.get(NYT_CONNECTIONS_URL)
        data = json.loads(response.text)
        
        # Convert to dataset format
        dataset_rows = []
        for game in data:
            game_state = NYTGameState.initialize(game['answers'])
            
            # Format answer as the list of groups
            dataset_rows.append({
                'question': game_state.get_current_prompt(),
                'answer': json.dumps(game_state.model_dump())
            })
            
        # Set seed for reproducibility
        random.seed(RANDOM_SEED)
        
        # Shuffle and split into train/eval
        random.shuffle(dataset_rows)
        
        print(f"Total dataset size: {len(dataset_rows)}")
        
        # Create datasets
        train_rows = dataset_rows[:-num_eval_samples]
        eval_rows = dataset_rows[-num_eval_samples:]
        
        train_dataset = Dataset.from_list(train_rows)
        eval_dataset = Dataset.from_list(eval_rows)
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Evaluation dataset size: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset