#!/usr/bin/env python3
"""
ECG Audio Analysis CLI
High-performance audio analysis pipeline for dynamic subtitle generation

Main entry point for the ECG Audio Analysis system with comprehensive CLI interface.
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional
import time

import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline.manager import PipelineManager
# from src.services.result_synthesizer import ResultSynthesizer, SynthesisInput  # Currently unused in MVP
from src.utils.logger import setup_logging, get_logger
from config.base_settings import BaseConfig, ProcessingConfig, ValidationConfig
from config.aws_settings import AWSConfig
from config.model_configs import SpeakerDiarizationConfig
# from src.models.output_models import ModelVersions  # Currently unused in MVP

# Initialize CLI app
app = typer.Typer(
    name="ecg-audio-analyzer",
    help="High-performance audio analysis pipeline for dynamic subtitle generation",
    add_completion=False,
)

console = Console()


def setup_configs(
    gpu: bool = True,
    workers: int = 4,
    aws_instance: str = "p3.2xlarge",
    enable_cloudwatch: bool = False,
) -> tuple:
    """Setup configuration objects"""

    # Base configuration
    base_config = BaseConfig(max_workers=workers, enable_cleanup=True, log_level="INFO")

    # Processing configuration
    processing_config = ProcessingConfig(
        enable_multiprocessing=True,
        enable_async_io=True,
        enable_memory_monitoring=True,
        track_processing_time=True,
    )

    # Validation configuration
    validation_config = ValidationConfig(
        enable_quality_checks=True,
        min_duration_seconds=1.0,
        max_duration_seconds=3600,  # 1 hour max
    )

    # AWS configuration (optional)
    aws_config = None
    if gpu or aws_instance != "local":
        aws_config = AWSConfig(
            instance_type=aws_instance,
            cuda_device="cuda:0" if gpu else "cpu",
            enable_gpu_monitoring=gpu,
            concurrent_workers=workers,
        )

    # Speaker diarization configuration
    speaker_config = SpeakerDiarizationConfig(
        device="cuda:0" if gpu else "cpu", enable_fp16=gpu, batch_size=8 if gpu else 4
    )

    return base_config, processing_config, validation_config, aws_config, speaker_config


def display_system_info(gpu: bool = True):
    """Display system information and capabilities"""

    system_info = Table(title="System Information")
    system_info.add_column("Component", style="cyan", no_wrap=True)
    system_info.add_column("Status", style="magenta")
    system_info.add_column("Details", style="green")

    # Check GPU availability
    gpu_status = "❌ Not Available"
    gpu_details = "No CUDA support"

    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = (
                torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            )
            gpu_status = "✅ Available"
            gpu_details = f"{device_count} device(s): {device_name}"
    except ImportError:
        gpu_details = "PyTorch not installed"

    system_info.add_row("GPU Acceleration", gpu_status, gpu_details)

    # Check required dependencies
    dependencies = [
        ("FFmpeg", "ffmpeg"),
        ("yt-dlp", "yt_dlp"),
        ("librosa", "librosa"),
        ("pyannote.audio", "pyannote.audio"),
    ]

    for name, module in dependencies:
        try:
            __import__(module)
            system_info.add_row(name, "✅ Installed", "Ready")
        except ImportError:
            system_info.add_row(name, "❌ Missing", "Required for full functionality")

    console.print(system_info)


def display_progress_summary(results: List, total_time: float):
    """Display processing results summary"""

    successful = [
        r for r in results if hasattr(r, "metadata") and r.metadata.total_speakers > 0
    ]
    failed = len(results) - len(successful)

    summary = Table(title="Processing Summary")
    summary.add_column("Metric", style="cyan", no_wrap=True)
    summary.add_column("Value", style="magenta", justify="right")

    summary.add_row("Total Files", str(len(results)))
    summary.add_row("Successful", str(len(successful)))
    summary.add_row("Failed", str(failed))
    summary.add_row("Total Time", f"{total_time:.1f}s")
    summary.add_row(
        "Avg Time per File", f"{total_time/len(results):.1f}s" if results else "N/A"
    )

    if successful:
        total_speakers = sum(r.metadata.total_speakers for r in successful)
        total_segments = sum(len(r.timeline) for r in successful)
        total_duration = sum(r.metadata.duration for r in successful)

        summary.add_row("Total Speakers", str(total_speakers))
        summary.add_row("Total Segments", str(total_segments))
        summary.add_row("Total Audio Duration", f"{total_duration:.1f}s")
        summary.add_row(
            "Processing Speed",
            f"{total_duration/total_time:.1f}x realtime" if total_time > 0 else "N/A",
        )

    console.print(summary)

    # Show individual file results if there are failures
    if failed > 0:
        console.print("\n[red]Failed Files:[/red]")
        for i, result in enumerate(results):
            if not hasattr(result, "metadata") or result.metadata.total_speakers == 0:
                console.print(f"  • File {i+1}: Processing failed")


@app.command()
def analyze(
    source: List[str] = typer.Argument(
        ..., help="Audio/video files or URLs to analyze"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for results"
    ),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Enable GPU acceleration"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of worker threads"),
    compress: bool = typer.Option(
        True, "--compress/--no-compress", help="Compress output JSON files"
    ),
    prettify: bool = typer.Option(
        True, "--prettify/--no-prettify", help="Format JSON output"
    ),
    aws_instance: str = typer.Option(
        "p3.2xlarge", "--aws-instance", help="AWS instance type for optimization"
    ),
    enable_cloudwatch: bool = typer.Option(
        False, "--cloudwatch", help="Enable CloudWatch logging"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    show_progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Show progress bars"
    ),
):
    """
    Analyze audio/video files for speaker diarization and acoustic features.

    Supports various input formats including MP4, WAV, MP3, and YouTube URLs.
    Results are saved as comprehensive JSON metadata files.
    """

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(
        level=log_level,
        enable_performance_monitoring=True,
        enable_cloudwatch=enable_cloudwatch,
    )
    logger = get_logger()

    # Display startup information
    console.print(
        Panel.fit(
            "[bold blue]ECG Audio Analysis Pipeline[/bold blue]\n"
            "High-performance audio analysis for dynamic subtitle generation",
            title="🎵 Starting Analysis",
        )
    )

    if verbose:
        display_system_info(gpu)

    # Setup configurations
    configs = setup_configs(
        gpu=gpu,
        workers=workers,
        aws_instance=aws_instance,
        enable_cloudwatch=enable_cloudwatch,
    )
    base_config, processing_config, validation_config, aws_config, speaker_config = (
        configs
    )

    # Create output directory
    if output is None:
        output = Path("./output")
    output.mkdir(parents=True, exist_ok=True)

    # Validate sources
    valid_sources = []
    for src in source:
        if src.startswith(("http://", "https://")):
            valid_sources.append(src)  # URL
        else:
            src_path = Path(src)
            if src_path.exists():
                valid_sources.append(src_path)
            else:
                console.print(f"[red]Warning:[/red] File not found: {src}")

    if not valid_sources:
        console.print("[red]Error:[/red] No valid sources found")
        raise typer.Exit(1)

    console.print(f"Processing {len(valid_sources)} file(s)...")

    async def run_analysis():
        """Run the async analysis pipeline"""

        # Initialize pipeline manager
        with PipelineManager(
            base_config=base_config,
            processing_config=processing_config,
            aws_config=aws_config,
            speaker_config=speaker_config,
        ) as pipeline:

            # Initialize result synthesizer (currently unused in MVP)
            # synthesizer = ResultSynthesizer(
            #     enable_compression=compress,
            #     compression_threshold_mb=5.0,
            #     enable_validation=True,
            # )

            results = []
            start_time = time.time()

            # Process files with progress tracking
            if show_progress and len(valid_sources) > 1:
                # Batch processing with progress bar
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:

                    task = progress.add_task(
                        "Processing files...", total=len(valid_sources)
                    )

                    # Process each file individually for better progress tracking
                    for i, src in enumerate(valid_sources):
                        progress.update(
                            task, description=f"Processing {Path(str(src)).name}"
                        )

                        try:
                            # Process single file
                            result = await pipeline.process_single(src)

                            # Create output filename
                            src_name = Path(str(src)).stem
                            output_file = output / f"{src_name}_analysis.json"

                            # Create synthesis input (currently unused in MVP)
                            # synthesis_input = SynthesisInput(
                            #     filename=Path(str(src)).name,
                            #     duration=result.metadata.duration,
                            #     audio_extraction_result=None,  # Will be mock for this version
                            #     diarization_result=None,  # Will be mock for this version
                            #     processing_start_time=start_time,
                            #     gpu_acceleration_used=gpu,
                            #     model_versions=ModelVersions(),
                            # )

                            # Export result
                            if (
                                compress
                                and output_file.stat().st_size > 5 * 1024 * 1024
                            ):  # 5MB threshold
                                output_file = output_file.with_suffix(".json.gz")

                            with open(output_file, "w") as f:
                                f.write(
                                    result.model_dump_json(
                                        indent=2 if prettify else None
                                    )
                                )

                            results.append(result)

                            logger.info(
                                "file_processed",
                                source=str(src),
                                output=str(output_file),
                                speakers=result.metadata.total_speakers,
                                duration=result.metadata.duration,
                            )

                        except Exception as e:
                            logger.error(
                                "file_processing_failed", source=str(src), error=str(e)
                            )
                            # Add empty result for failed processing
                            from src.models.output_models import (
                                create_empty_analysis_result,
                            )

                            results.append(
                                create_empty_analysis_result(Path(str(src)).name, 0.0)
                            )

                        progress.advance(task)

            else:
                # Single file or no progress bar
                if len(valid_sources) == 1:
                    console.print(f"Processing: [cyan]{valid_sources[0]}[/cyan]")

                for src in valid_sources:
                    try:
                        result = await pipeline.process_single(src)

                        # Create output filename
                        src_name = Path(str(src)).stem
                        output_file = output / f"{src_name}_analysis.json"

                        # Export result
                        with open(output_file, "w") as f:
                            f.write(
                                result.model_dump_json(indent=2 if prettify else None)
                            )

                        results.append(result)

                        console.print(f"✅ Completed: [green]{output_file}[/green]")

                    except Exception as e:
                        logger.error(
                            "file_processing_failed", source=str(src), error=str(e)
                        )
                        from src.models.output_models import (
                            create_empty_analysis_result,
                        )

                        results.append(
                            create_empty_analysis_result(Path(str(src)).name, 0.0)
                        )
                        console.print(f"❌ Failed: [red]{src}[/red] - {str(e)}")

            total_time = time.time() - start_time

            # Display results summary
            console.print("\n")
            display_progress_summary(results, total_time)

            return results

    # Run the async pipeline
    try:
        results = asyncio.run(run_analysis())
        console.print(
            f"\n[green]✨ Analysis completed![/green] Results saved to: [cyan]{output}[/cyan]"
        )

        # Show first result sample if available
        successful_results = [
            r for r in results if hasattr(r, "timeline") and r.timeline
        ]
        if successful_results:
            sample = successful_results[0]
            console.print(f"\n[bold]Sample Result:[/bold] {sample.metadata.filename}")
            console.print(f"  • Duration: {sample.metadata.duration:.1f}s")
            console.print(f"  • Speakers: {sample.metadata.total_speakers}")
            console.print(f"  • Segments: {len(sample.timeline)}")
            console.print(f"  • Processing Time: {sample.metadata.processing_time}")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Analysis interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ Analysis failed: {str(e)}[/red]")
        logger.error("analysis_failed", error=str(e))
        raise typer.Exit(1)


@app.command()
def info():
    """Display system information and check dependencies"""
    display_system_info(True)


@app.command()
def test():
    """Run a quick test with sample data"""
    console.print("[yellow]Test mode - this would process a sample file[/yellow]")
    console.print("Use the 'analyze' command with real audio/video files")


@app.command()
def version():
    """Show version information"""
    console.print("ECG Audio Analyzer v1.0.0")
    console.print(
        "High-performance audio analysis pipeline for dynamic subtitle generation"
    )


if __name__ == "__main__":
    # Add version option
    def version_callback(value: bool):
        if value:
            console.print("ECG Audio Analyzer v1.0.0")
            raise typer.Exit()

    # Setup global options
    @app.callback()
    def main(
        version: Optional[bool] = typer.Option(
            None, "--version", callback=version_callback, is_eager=True
        )
    ):
        """
        ECG Audio Analysis Pipeline

        High-performance backend analysis engine for dynamic subtitle generation.
        Extracts audio features, performs speaker diarization, and generates
        comprehensive metadata for subtitle styling.
        """
        pass

    app()
