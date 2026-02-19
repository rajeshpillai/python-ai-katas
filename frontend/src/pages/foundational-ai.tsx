import type { ParentComponent } from "solid-js";
import MainLayout from "../components/layout/main-layout";
import "./foundational-ai.css";

const FoundationalAi: ParentComponent = (props) => {
  return (
    <MainLayout>
      {props.children}
    </MainLayout>
  );
};

export default FoundationalAi;
