import type { ParentComponent } from "solid-js";
import Sidebar from "./sidebar";
import ThemeToggle from "./theme-toggle";
import "./main-layout.css";

const MainLayout: ParentComponent<{ trackId: string }> = (props) => {
  return (
    <div class="main-layout">
      <Sidebar trackId={props.trackId} />
      <div class="main-layout__content">
        <header class="main-layout__top-bar">
          <ThemeToggle />
        </header>
        <main class="main-layout__main">{props.children}</main>
      </div>
    </div>
  );
};

export default MainLayout;
