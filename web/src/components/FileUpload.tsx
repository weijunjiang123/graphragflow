import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, FileText, AlertCircle, CheckCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Progress } from '@/components/ui/Progress';
import { useFileUpload } from '@/hooks';
import { formatFileSize, isValidFileType } from '@/lib/utils';

interface FileUploadProps {
  onUploadComplete?: (result: any) => void;
  onUploadError?: (error: Error) => void;
}

const ALLOWED_FILE_TYPES = [
  '.pdf',
  '.txt',
  '.doc',
  '.docx',
  '.md',
  'text/plain',
  'application/pdf',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
];

export function FileUpload({ onUploadComplete, onUploadError }: FileUploadProps) {
  const { uploadFile, isUploading } = useFileUpload();
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = Array.from(e.dataTransfer.files);
    await handleFiles(files);
  }, [uploadFile, onUploadComplete, onUploadError]);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      await handleFiles(files);
      e.target.value = ''; // Reset input
    }
  }, [uploadFile, onUploadComplete, onUploadError]);

  const handleFiles = useCallback(async (files: File[]) => {
    for (const file of files) {
      if (!isValidFileType(file, ALLOWED_FILE_TYPES)) {
        onUploadError?.(new Error(`不支持的文件类型: ${file.name}`));
        continue;
      }

      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        onUploadError?.(new Error(`文件过大: ${file.name} (最大 10MB)`));
        continue;
      }

      try {
        setUploadProgress(0);
        const result = await uploadFile(file);
        setUploadProgress(100); // Set progress to 100% when complete
        setUploadedFiles(prev => [...prev, { 
          ...result, 
          filename: file.name, 
          size: file.size,
          status: 'completed'
        }]);
        onUploadComplete?.(result);
        setUploadProgress(0);
      } catch (error) {
        console.error('File upload failed:', error);
        setUploadedFiles(prev => [...prev, { 
          filename: file.name, 
          size: file.size,
          status: 'failed',
          error: error instanceof Error ? error.message : 'Upload failed'
        }]);
        onUploadError?.(error as Error);
        setUploadProgress(0);
      }
    }
  }, [uploadFile, onUploadComplete, onUploadError]);

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="w-5 h-5" />
          文档上传
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div
          className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${
            dragActive
              ? 'border-blue-500 bg-blue-500/10'
              : 'border-white/20 hover:border-white/40'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            multiple
            accept={ALLOWED_FILE_TYPES.join(',')}
            onChange={handleFileSelect}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            disabled={isUploading}
          />
          
          <div className="space-y-4">
            <div className="w-16 h-16 mx-auto bg-white/10 rounded-full flex items-center justify-center">
              <Upload className="w-8 h-8 text-white/70" />
            </div>
            
            <div>
              <p className="text-white text-lg font-medium">
                拖拽文件到此处或点击选择
              </p>
              <p className="text-white/70 text-sm mt-2">
                支持 PDF, TXT, DOC, DOCX, MD 格式，最大 10MB
              </p>
            </div>
            
            <Button variant="secondary" disabled={isUploading}>
              选择文件
            </Button>
          </div>
        </div>

        {isUploading && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-2"
          >
            <div className="flex justify-between text-sm">
              <span className="text-white/70">上传中...</span>
              <span className="text-white">{uploadProgress}%</span>
            </div>
            <Progress value={uploadProgress} />
          </motion.div>
        )}

        <AnimatePresence>
          {uploadedFiles.map((file, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/10"
            >
              <div className="flex items-center gap-3">
                <FileText className="w-5 h-5 text-blue-400" />
                <div>
                  <p className="text-white text-sm font-medium">{file.filename}</p>
                  <p className="text-white/70 text-xs">
                    {formatFileSize(file.size)} • 任务ID: {file.task_id}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                {file.status === 'completed' ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : file.status === 'failed' ? (
                  <AlertCircle className="w-5 h-5 text-red-400" />
                ) : (
                  <div className="w-5 h-5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                )}
                
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => removeFile(index)}
                  className="w-8 h-8"
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </CardContent>
    </Card>
  );
}
