export const ROUTES = {
  LANDING: "/",
  LANGUAGE: "/:lang",
  FOUNDATIONAL_AI: "/:lang/foundational-ai",
  KATA: "/:lang/foundational-ai/:phaseId/:kataId",
  TRADITIONAL_AI_ML: "/:lang/traditional-ai-ml",
  TRADITIONAL_KATA: "/:lang/traditional-ai-ml/:phaseId/:kataId",
} as const;
