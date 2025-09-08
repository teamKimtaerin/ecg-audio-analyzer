"""
ECG Audio Analyzer - Command Line Interface

Provides command-line access to the audio analysis functionality.
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

from .api import analyze_audio, AnalysisConfig

# Create Typer app
app = typer.Typer(
    name="ecg-analyze",
    help="🎵 ECG Audio Analyzer - High-performance audio analysis with speaker diarization and emotion detection",
    add_completion=False,
)

console = Console()


@app.command()
def analyze(
    input_path: str = typer.Argument(
        ..., help="Path to audio/video file or YouTube URL"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output JSON file path"
    ),
    gpu: bool = typer.Option(False, "--gpu", help="Enable GPU acceleration"),
    emotion_detection: bool = typer.Option(
        True, "--emotion-detection/--no-emotion", help="Enable emotion analysis"
    ),
    language: str = typer.Option("en", "--language", help="Language code (en, auto)"),
    optimize_subtitles: bool = typer.Option(
        True,
        "--optimize-subtitles/--no-optimize",
        help="Optimize for subtitle generation",
    ),
    workers: int = typer.Option(
        4, "--workers", help="Number of parallel workers for CPU processing"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Analyze audio file with speaker diarization and emotion detection.

    Examples:

        ecg-analyze video.mp4

        ecg-analyze video.mp4 --output results.json --gpu

        ecg-analyze "https://youtube.com/watch?v=..." --emotion-detection
    """

    # Create configuration
    config = AnalysisConfig(
        enable_gpu=gpu,
        emotion_detection=emotion_detection,
        language=language,
        optimize_for_subtitles=optimize_subtitles,
        max_workers=workers,
    )

    # Setup logging based on verbosity
    if verbose:
        import logging

        logging.basicConfig(level=logging.INFO)

    # Run analysis
    asyncio.run(_run_analysis(input_path, config, output, verbose))


async def _run_analysis(
    input_path: str, config: AnalysisConfig, output_path: Optional[str], verbose: bool
):
    """Run the analysis with progress indication"""

    try:
        input_file = Path(input_path)

        # Display initial info
        rprint("\n🎵 [bold blue]ECG Audio Analyzer[/bold blue]")
        rprint(f"📁 Input: [yellow]{input_path}[/yellow]")

        if input_file.exists():
            size_mb = input_file.stat().st_size / 1024 / 1024
            rprint(f"📊 Size: [green]{size_mb:.1f} MB[/green]")

        rprint(
            f"⚙️  GPU: [{'green' if config.enable_gpu else 'red'}]{'Enabled' if config.enable_gpu else 'Disabled'}[/{'green' if config.enable_gpu else 'red'}]"
        )
        rprint(
            f"🎭 Emotions: [{'green' if config.emotion_detection else 'red'}]{'Enabled' if config.emotion_detection else 'Disabled'}[/{'green' if config.emotion_detection else 'red'}]"
        )

        # Run analysis with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:

            task = progress.add_task("🔄 Analyzing audio...", total=None)

            result = await analyze_audio(input_path, config, output_path)

            progress.update(task, description="✅ Analysis complete!")

        # Display results
        _display_results(result, verbose)

        # Output file info - JSON is always saved now
        from .utils.output_manager import get_output_manager

        output_manager = get_output_manager()

        if output_path:
            rprint(f"\n💾 Results saved to: [green]{output_path}[/green]")
        else:
            # Show where the auto-generated file was saved
            auto_filename = output_manager.generate_output_filename(input_path)
            auto_path = output_manager.base_output_dir / auto_filename
            rprint(f"\n💾 Results auto-saved to: [green]{auto_path}[/green]")

        rprint("\n🎉 [bold green]Analysis completed successfully![/bold green]")

    except FileNotFoundError:
        rprint(f"❌ [red]Error: File not found: {input_path}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        rprint(f"❌ [red]Error: {str(e)}[/red]")
        if verbose:
            import traceback

            rprint(traceback.format_exc())
        raise typer.Exit(1)


def _display_results(result, verbose: bool):
    """Display analysis results in a formatted table"""

    # Summary table
    summary_table = Table(
        title="📊 Analysis Summary", show_header=True, header_style="bold magenta"
    )
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Duration", f"{result.duration:.1f}s")
    summary_table.add_row("Processing Time", f"{result.processing_time:.1f}s")
    summary_table.add_row(
        "Speed Ratio", f"{result.duration/result.processing_time:.1f}x"
    )
    summary_table.add_row("Total Segments", str(result.total_segments))
    summary_table.add_row("Unique Speakers", str(result.unique_speakers))
    summary_table.add_row("Dominant Emotion", result.dominant_emotion or "N/A")
    summary_table.add_row("Avg Confidence", f"{result.avg_confidence:.2f}")

    console.print()
    console.print(summary_table)

    # Speaker breakdown
    if result.speakers:
        speaker_table = Table(
            title="👥 Speaker Breakdown", show_header=True, header_style="bold blue"
        )
        speaker_table.add_column("Speaker ID", style="cyan")
        speaker_table.add_column("Duration", style="green")
        speaker_table.add_column("Segments", style="yellow")
        speaker_table.add_column("Confidence", style="magenta")
        speaker_table.add_column("Emotions", style="red")

        for speaker_id, info in result.speakers.items():
            emotions = (
                ", ".join(info.get("emotions", [])) if info.get("emotions") else "N/A"
            )
            speaker_table.add_row(
                speaker_id,
                f"{info['total_duration']:.1f}s",
                str(info["segment_count"]),
                f"{info['avg_confidence']:.2f}",
                emotions,
            )

        console.print()
        console.print(speaker_table)

    # Detailed segments (if verbose)
    if verbose and result.segments:
        segments_table = Table(
            title="🎯 Segment Details", show_header=True, header_style="bold yellow"
        )
        segments_table.add_column("Time", style="cyan")
        segments_table.add_column("Speaker", style="green")
        segments_table.add_column("Emotion", style="red")
        segments_table.add_column("Confidence", style="magenta")

        for segment in result.segments[:10]:  # Show first 10 segments
            time_range = f"{segment.start_time:.1f}-{segment.end_time:.1f}s"
            emotion = segment.emotion.emotion if segment.emotion else "N/A"
            confidence = f"{segment.speaker.confidence:.2f}"

            segments_table.add_row(
                time_range, segment.speaker.speaker_id, emotion, confidence
            )

        if len(result.segments) > 10:
            segments_table.add_row("...", "...", "...", "...")
            segments_table.add_row(
                f"({len(result.segments) - 10} more segments)", "", "", ""
            )

        console.print()
        console.print(segments_table)


@app.command()
def version():
    """Show version information"""
    from . import __version__, VERSION_INFO, get_system_info

    rprint(f"\n🎵 [bold blue]ECG Audio Analyzer[/bold blue] v{__version__}")

    # Version info table
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Key", style="cyan")
    info_table.add_column("Value", style="green")

    info_table.add_row("Version", __version__)
    info_table.add_row("Python Required", VERSION_INFO["python_requires"])
    info_table.add_row(
        "Supported Formats", ", ".join(VERSION_INFO["supported_formats"])
    )

    console.print()
    console.print(info_table)

    # System info
    sys_info = get_system_info()
    rprint("\n💻 [bold]System Information[/bold]")
    rprint(f"Python: [green]{sys_info['python_version'].split()[0]}[/green]")
    rprint(f"Platform: [green]{sys_info['platform']}[/green]")
    rprint(
        f"GPU Available: [{'green' if sys_info['gpu_available'] else 'red'}]{sys_info['gpu_available']}[/{'green' if sys_info['gpu_available'] else 'red'}]"
    )

    if sys_info["gpu_count"] > 0:
        rprint(f"GPU Count: [green]{sys_info['gpu_count']}[/green]")


@app.command()
def test():
    """Run a quick test to verify installation"""
    rprint("\n🧪 [bold blue]Testing ECG Audio Analyzer Installation[/bold blue]")

    try:
        # Test imports
        rprint("📦 Testing imports... ", end="")
        from . import get_version, get_system_info

        rprint("[green]✅ OK[/green]")

        # Test configuration
        rprint("⚙️  Testing configuration... ", end="")
        config = AnalysisConfig()
        rprint("[green]✅ OK[/green]")

        # Test system info
        rprint("💻 Testing system detection... ", end="")
        info = get_system_info()
        rprint("[green]✅ OK[/green]")

        rprint("\n🎉 [bold green]Installation test passed![/bold green]")
        rprint(f"ECG Audio Analyzer v{get_version()} is ready to use.")

        if not info["gpu_available"]:
            rprint("ℹ️  [yellow]Note: GPU not available, will use CPU mode[/yellow]")

    except Exception as e:
        rprint("[red]❌ FAILED[/red]")
        rprint(f"Error: {str(e)}")
        raise typer.Exit(1)


def main():
    """Main entry point for CLI"""
    app()


if __name__ == "__main__":
    main()
