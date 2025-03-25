import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Track progress across different stages of processing"""
    
    def __init__(self, total_stages: int):
        """Initialize progress tracker
        
        Args:
            total_stages: Total number of stages to track
        """
        self.total_stages = total_stages
        self.current_stage = 0
        self.start_time = time.time()
        self.stage_times = []
        
    def update(self, description: str) -> None:
        """Update progress to the next stage
        
        Args:
            description: Description of the current stage
        """
        stage_time = time.time()
        
        # Calculate elapsed time for the previous stage if not the first stage
        if self.stage_times:
            prev_stage_elapsed = stage_time - self.stage_times[-1]
        else:
            prev_stage_elapsed = stage_time - self.start_time
            
        self.stage_times.append(stage_time)
        self.current_stage += 1
        
        elapsed_total = stage_time - self.start_time
        
        # Log stage information
        logger.info(f"[{self.current_stage}/{self.total_stages}] {description} - "
                   f"Stage: {prev_stage_elapsed:.2f}s - Total: {elapsed_total:.2f}s")
        
        print(f"Progress: {self.current_stage}/{self.total_stages} - {description} "
              f"({prev_stage_elapsed:.2f}s)")
        
    def summary(self) -> str:
        """Generate a summary of all stages
        
        Returns:
            Summary string
        """
        total_time = time.time() - self.start_time
        
        summary_lines = [f"Total processing time: {total_time:.2f} seconds"]
        
        # Add individual stage times if available
        if len(self.stage_times) > 1:
            summary_lines.append("\nStage times:")
            
            for i in range(1, len(self.stage_times)):
                stage_time = self.stage_times[i] - self.stage_times[i-1]
                summary_lines.append(f"  Stage {i}: {stage_time:.2f}s")
                
        return "\n".join(summary_lines)
        
    def reset(self) -> None:
        """Reset the tracker"""
        self.current_stage = 0
        self.start_time = time.time()
        self.stage_times = []
