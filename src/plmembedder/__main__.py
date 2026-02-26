# plmembedder/__main__.py
"""Command-line interface for PLMEmbedder."""

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    PipelineConfig,
    EmbedderConfig,
    CacheConfig,
    DecoyConfig,
    OutputConfig,
    EmbedderType,
    DecoyType,
)
from .pipeline import EmbeddingPipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog='plmembedder',
        description='Protein Language Model Embedding Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic embedding
  plmembedder proteins.fasta -o results/

  # With caching
  plmembedder proteins.fasta --cache-embeddings -c cache/ -o results/

  # Use cached embeddings
  plmembedder proteins.fasta --cache-embeddings -c cache/ -o results2/

  # Generate decoys
  plmembedder proteins.fasta --n-decoys 1 --decoy-type reversed -o results/

  # Use smaller/faster model
  plmembedder proteins.fasta --model esm2_t6_8M_UR50D -o results/

  # CPU only
  plmembedder proteins.fasta --device cpu -o results/
"""
    )

    # Required arguments
    parser.add_argument(
        'fasta',
        type=str,
        help='Input FASTA file'
    )

    # Output options
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )

    # Model options
    parser.add_argument(
        '--model',
        type=str,
        default='esm2_t33_650M_UR50D',
        help='Model name (default: esm2_t33_650M_UR50D)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['esm2', 'esm1b', 'protbert', 'prott5'],
        default='esm2',
        help='Model type (default: esm2)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use: cuda or cpu (default: cuda)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for embedding (default: 4)'
    )
    parser.add_argument(
        '--layer',
        type=int,
        default=-1,
        help='Model layer to extract embeddings from (default: -1, last layer)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=1024,
        help='Maximum sequence length (default: 1024)'
    )

    # Cache options
    parser.add_argument(
        '--cache-embeddings',
        action='store_true',
        help='Enable embedding caching'
    )
    parser.add_argument(
        '-c', '--cache-dir',
        type=str,
        default='embeddings_cache',
        help='Cache directory (default: embeddings_cache)'
    )

    # Decoy options
    parser.add_argument(
        '--n-decoys',
        type=int,
        default=0,
        help='Number of decoys per sequence (default: 0, disabled)'
    )
    parser.add_argument(
        '--decoy-type',
        type=str,
        choices=['reversed', 'permuted', 'shuffled_blocks'],
        default='reversed',
        help='Decoy generation method (default: reversed)'
    )
    parser.add_argument(
        '--decoy-prefix',
        type=str,
        default='DECOY_',
        help='Prefix for decoy sequence IDs (default: DECOY_)'
    )
    parser.add_argument(
        '--decoy-seed',
        type=int,
        default=None,
        help='Random seed for decoy generation'
    )
    parser.add_argument(
        '--decoys-only',
        action='store_true',
        help='Embed only decoy sequences (exclude originals)'
    )
    parser.add_argument(
        '--write-decoy-fasta',
        type=str,
        default=None,
        help='Write decoy sequences to FASTA file'
    )

    # Input options
    parser.add_argument(
        '--max-sequences',
        type=int,
        default=None,
        help='Maximum number of sequences to process'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip sequence validation'
    )

    # Other options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--embed-only',
        action='store_true',
        help='Only compute embeddings, skip saving consolidated output'
    )

    return parser


def main(args=None):
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(args)

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Build configuration
    embedder_config = EmbedderConfig(
        embedder_type=EmbedderType(args.model_type),
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        layer=args.layer,
        max_sequence_length=args.max_length,
    )

    cache_config = CacheConfig(
        enabled=args.cache_embeddings,
        cache_dir=args.cache_dir,
    )

    decoy_config = DecoyConfig(
        n_decoys=args.n_decoys,
        decoy_type=DecoyType(args.decoy_type),
        decoy_prefix=args.decoy_prefix,
        random_seed=args.decoy_seed,
        decoys_only=args.decoys_only,
    )

    output_config = OutputConfig(
        output_dir=args.output,
        save_embeddings=not args.embed_only,
    )

    config = PipelineConfig(
        embedder=embedder_config,
        cache=cache_config,
        decoy=decoy_config,
        output=output_config,
        max_sequences=args.max_sequences,
        validate_sequences=not args.no_validate,
    )

    # Run pipeline
    try:
        pipeline = EmbeddingPipeline(config)
        embeddings = pipeline.run(args.fasta, args.output)

        # Write decoy FASTA if requested
        if args.write_decoy_fasta and pipeline.decoy_results:
            from .io.fasta_parser import FastaParser
            decoy_seqs = [
                (d.decoy_id, d.decoy_sequence)
                for d in pipeline.decoy_results
            ]
            FastaParser.write(decoy_seqs, args.write_decoy_fasta)
            logger.info(f"Wrote {len(decoy_seqs)} decoys to {args.write_decoy_fasta}")

        logger.info(f"Successfully embedded {len(embeddings)} sequences")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()
