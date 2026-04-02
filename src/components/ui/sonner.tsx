import { useTheme } from "next-themes";
import { Toaster as Sonner, toast } from "sonner";

type ToasterProps = React.ComponentProps<typeof Sonner>;

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme = "system" } = useTheme();

  return (
    <Sonner
      position="bottom-right"
      visibleToasts={4}
      closeButton
      expand
      theme={theme as ToasterProps["theme"]}
      className="toaster group"
      toastOptions={{
        classNames: {
          toast:
            "group toast rounded-xl border border-border/80 shadow-xl backdrop-blur-sm group-[.toaster]:bg-card/95 group-[.toaster]:text-foreground data-[visible=true]:animate-in data-[visible=true]:fade-in-0 data-[visible=true]:slide-in-from-bottom-3 data-[visible=false]:animate-out data-[visible=false]:fade-out-0 data-[visible=false]:slide-out-to-bottom-2",
          description: "group-[.toast]:text-muted-foreground",
          actionButton: "group-[.toast]:bg-primary group-[.toast]:text-primary-foreground",
          cancelButton: "group-[.toast]:bg-muted group-[.toast]:text-muted-foreground",
          success: "!bg-success/15 !border-success/50 !text-green-200",
          error: "!bg-destructive/20 !border-destructive/60 !text-red-200",
        },
      }}
      {...props}
    />
  );
};

export { Toaster, toast };
