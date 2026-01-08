import { setup } from "@twind/core";
import presetAutoprefix from "@twind/preset-autoprefix";
import presetTailwind from "@twind/preset-tailwind";

setup({
  presets: [presetAutoprefix(), presetTailwind()]
});
