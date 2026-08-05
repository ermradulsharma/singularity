def dfa_simulate(states: set, alphabet: set, transition_function: dict, start_state: str, accept_states: set, string: str) -> bool:
    """Theory of computation: DFA (Deterministic Finite Automaton) simulator"""
    current_state = start_state
    for char in string:
        if char not in alphabet:
            return False
        current_state = transition_function.get((current_state, char), None)
        if current_state is None:
            return False
    return current_state in accept_states

def lexical_analysis(source_code: str) -> list:
    """Compiler Design: Lexical Analyzer"""
    tokens = []
    for word in source_code.split():
        if word in ['if', 'else', 'while', 'for']:
            tokens.append(('KEYWORD', word))
        elif word.isdigit():
            tokens.append(('NUMBER', word))
        else:
            tokens.append(('IDENTIFIER', word))
    return tokens
