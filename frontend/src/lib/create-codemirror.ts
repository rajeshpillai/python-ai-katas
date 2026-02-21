import { createSignal, onMount, onCleanup, Accessor } from "solid-js";
import { EditorState, Extension } from "@codemirror/state";
import { EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter } from "@codemirror/view";
import { defaultKeymap, indentWithTab, history, historyKeymap } from "@codemirror/commands";
import { python } from "@codemirror/lang-python";
import { bracketMatching, indentOnInput, foldGutter, foldKeymap } from "@codemirror/language";
import { autocompletion, closeBrackets, closeBracketsKeymap, completionKeymap } from "@codemirror/autocomplete";
import { kataTheme } from "./codemirror-theme";

interface CodeMirrorOptions {
  code: Accessor<string>;
  onCodeChange: (code: string) => void;
  onCtrlEnter?: () => void;
}

export function createCodeMirror(options: CodeMirrorOptions) {
  let container: HTMLDivElement | undefined;
  let view: EditorView | undefined;
  let suppressUpdate = false;

  const extensions: Extension[] = [
    lineNumbers(),
    highlightActiveLine(),
    highlightActiveLineGutter(),
    history(),
    bracketMatching(),
    closeBrackets(),
    indentOnInput(),
    foldGutter(),
    autocompletion(),
    python(),
    kataTheme,
    keymap.of([
      ...defaultKeymap,
      ...historyKeymap,
      ...closeBracketsKeymap,
      ...foldKeymap,
      ...completionKeymap,
      indentWithTab,
      ...(options.onCtrlEnter
        ? [{
            key: "Ctrl-Enter",
            mac: "Cmd-Enter",
            run: () => { options.onCtrlEnter!(); return true; },
          }]
        : []),
    ]),
    EditorView.updateListener.of((update) => {
      if (update.docChanged && !suppressUpdate) {
        options.onCodeChange(update.state.doc.toString());
      }
    }),
  ];

  const ref = (el: HTMLDivElement) => {
    container = el;
  };

  onMount(() => {
    if (!container) return;
    view = new EditorView({
      state: EditorState.create({
        doc: options.code(),
        extensions,
      }),
      parent: container,
    });
  });

  onCleanup(() => {
    view?.destroy();
  });

  // Set code from outside (e.g. reset, kata change)
  const setCode = (newCode: string) => {
    if (!view) return;
    const current = view.state.doc.toString();
    if (current === newCode) return;
    suppressUpdate = true;
    view.dispatch({
      changes: { from: 0, to: current.length, insert: newCode },
    });
    suppressUpdate = false;
  };

  return { ref, setCode, getView: () => view };
}
