"""SearchSpaceComposer for combining and restricting search spaces."""

from __future__ import annotations

from orcalab.search_spaces.space import SearchSpace


class SearchSpaceComposer:
    """Static utilities for composing SearchSpace instances.

    Supports three composition patterns:

    - merge: union of two or more spaces (later spaces override on name conflict)
    - inherit: start from a parent space and selectively override with a child
    - restrict: project a space down to a named subset of its parameters
    """

    @staticmethod
    def merge(*spaces: SearchSpace, name: str) -> SearchSpace:
        """Merge one or more spaces into a new space by taking the union of their parameters.

        When multiple spaces define a parameter with the same name, the last space in the
        argument list wins. Conditions from all spaces are included in registration order.
        """
        merged = SearchSpace(name=name)
        for space in spaces:
            for param in space._params.values():
                merged._params[param.name] = param
        for space in spaces:
            merged._conditions.extend(space._conditions)
        return merged

    @staticmethod
    def inherit(parent: SearchSpace, child: SearchSpace) -> SearchSpace:
        """Create a new space that starts from the parent and overrides with child parameters.

        The resulting space takes its name and description from the child. Parameters
        defined in both parent and child use the child's definition. Conditions from both
        parent (first) and child are included.
        """
        result = SearchSpace(name=child.name, description=child.description)
        for param in parent._params.values():
            result._params[param.name] = param
        for param in child._params.values():
            result._params[param.name] = param
        result._conditions.extend(parent._conditions)
        result._conditions.extend(child._conditions)
        return result

    @staticmethod
    def restrict(space: SearchSpace, allowed_params: list[str]) -> SearchSpace:
        """Return a new space containing only the parameters whose names are in allowed_params.

        Conditions whose associated parameter is not in allowed_params are silently dropped.
        The resulting space has the same name and description as the input.
        """
        allowed = set(allowed_params)
        result = SearchSpace(name=space.name, description=space.description)
        for param in space._params.values():
            if param.name in allowed:
                result._params[param.name] = param
        for condition, param in space._conditions:
            if param.name in allowed:
                result._conditions.append((condition, param))
        return result
