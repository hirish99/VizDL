import { useEffect } from 'react';
import { fetchNodes } from '../api/client';
import { useGraphStore } from '../store/graphStore';

export function useNodeRegistry() {
  const setNodeDefinitions = useGraphStore((s) => s.setNodeDefinitions);
  const defs = useGraphStore((s) => s.nodeDefinitions);

  useEffect(() => {
    fetchNodes().then(setNodeDefinitions).catch(console.error);
  }, [setNodeDefinitions]);

  return defs;
}
