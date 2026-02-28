import type { ComponentType } from 'react';
import { BaseNode } from './BaseNode';

export function buildNodeTypes(
  nodeTypes: string[],
): Record<string, ComponentType<any>> {
  const result: Record<string, ComponentType<any>> = {};
  for (const nt of nodeTypes) {
    result[nt] = BaseNode;
  }
  return result;
}
