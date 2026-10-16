"""Single entry point: reproduce all numerical results, figures and CSV tables."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'code'))
if __name__=='__main__':
    from reproduce_pipeline import main
    main()
