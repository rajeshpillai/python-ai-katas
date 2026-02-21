import type { ParentComponent } from "solid-js";
import MainLayout from "../components/layout/main-layout";

const TraditionalAiMl: ParentComponent = (props) => {
  return (
    <MainLayout trackId="traditional-ai-ml">
      {props.children}
    </MainLayout>
  );
};

export default TraditionalAiMl;
