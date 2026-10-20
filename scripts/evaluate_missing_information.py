"""Run real extraction/GP judgment on the synthetic corpus, without treatment."""
import argparse
import asyncio
import json
from pathlib import Path

from src.agents.input import input_agent
from src.agents.general_physician import general_physician_agent


async def main(output):
    cases = json.loads(Path('data/cases/missing_information.json').read_text())
    results = []
    for case in cases:
        text = case['initial_input']
        rounds = []
        for supplement in [None, *case['supplements']]:
            if supplement:
                text += '\nAdditional patient information:\n' + supplement
            state = {'raw_input': text}
            state.update(await asyncio.to_thread(input_agent, state))
            extraction_errors = state.get('errors', [])
            assessment = await asyncio.to_thread(general_physician_agent, state)
            state.update(assessment)
            state['errors'] = extraction_errors + assessment.get('errors', [])
            rounds.append({'needs_more_information': state.get('needs_more_information'),
                           'questions': state.get('follow_up_questions', []),
                           'errors': state.get('errors', [])})
        passed = (rounds[0]['needs_more_information'] is True
                  and rounds[-1]['needs_more_information'] is False
                  and not any(item['errors'] for item in rounds))
        results.append({'id': case['id'], 'passed': passed, 'rounds': rounds})
        Path(output).write_text(json.dumps({'mode': 'real_model', 'results': results,
            'passed': sum(row['passed'] for row in results), 'total': len(results)}, ensure_ascii=False, indent=2))
        print(case['id'], 'PASS' if passed else 'FAIL', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/cases/model_evaluation.json')
    asyncio.run(main(parser.parse_args().output))
